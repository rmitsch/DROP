"""
Analysis of perceptual embedding quality.
Includes engineering of (visual quality criterion) features and building a model to predict perceived embedding quality.
"""

# todo - next steps:
#  - feature selection
#  - model training and optimization
#  - feature importance analysis
#     - use explainer/surrogate models - SHAP, skope-rules?
import os
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Dict, List, Tuple, Optional

import hdbscan
import shap
from sklearn import linear_model
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, \
    explained_variance_score
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import OneClassSVM
from tqdm import tqdm
import numpy as np
import pandas as pd
import tables
from tables import File, open_file
from scipy.spatial.distance import cdist
from source.analysis.dr_measure_correlations_to_user_score import read_user_scores
from data_generation.datasets import InputDataset
from generate_data import generate_instance
import plotly.express as px
import matplotlib.pyplot as plt
import lightgbm as lgbm
import optuna.integration.lightgbm as optuna_lgbm


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def compute_clusters(embedding: np.ndarray, **kwargs) -> Tuple[str]:
    """
    Compute clusters in an embedding.
    :param embedding: Embedding coordinates.
    :return: Tuple of cluster IDs, one per record in embedding. Note that -1 indicates no cluster association, records
    should hence not be treated as cluster.
    """

    return tuple([
        str(label) for label in hdbscan.HDBSCAN(**kwargs).fit(embedding).labels_
    ])


def engineer_features(data: Dict, user_scores: pd.DataFrame):
    """
    Engineers of (visual quality criterion) features and building a model to predict perceived embedding
    quality.
    Trains and evaluates model.
    :param data: Dictionary with original and embedding data per dataset name.
    :param user_scores: User scores for all embeddings and datasets, with dataset name as attribute.
    :parma user_scores: Obtained user scores for all embeddings.
    """

    training_data: pd.DataFrame = user_scores.copy(deep=True)
    embeddings: dict = {ds_name: {} for ds_name in datasets}

    #############################################
    # 1. Load embedding data from file.
    #############################################

    print("Loading embedding data.")
    pbar: tqdm = tqdm(total=len(training_data))
    for ix, row in training_data.iterrows():
        ds_name, emb_id = ix

        table: tables.group.Group = data[ds_name]["embedding_data"].root.projection_coordinates
        embeddings[ds_name][emb_id] = table["model" + str(emb_id)].read()
        orig_data: InputDataset = data[ds_name]["hd_dataset"]

        # Same number of rows in features and labels?
        assert orig_data.features().shape[0] == len(orig_data.labels())
        # Same number of rows in embedding as in original dataset?
        assert embeddings[ds_name][emb_id].shape[0] == len(orig_data.labels())

        pbar.update(1)
    pbar.close()

    # Close .h5 files.
    for ds_name in datasets:
        datasets[ds_name]["embedding_data"].close()

    #############################################
    # 2. Engineer (visual quality criterion)
    # features for every embedding.
    #############################################

    print("Computing features.")
    pbar = tqdm(total=len(training_data))
    features: List[Dict] = []
    visualize: bool = False
    for ix, row in training_data.iterrows():
        ds_name, emb_id = ix
        ld_cluster_ids: Tuple = compute_clusters(embeddings[ds_name][emb_id], metric="euclidean", min_samples=15)
        emb: np.ndarray = embeddings[ds_name][emb_id]
        # Extend by second dimension, if only 1D.
        if emb.shape[1] == 1:
            emb = np.asarray([emb[:, 0], [0] * len(emb)]).T

        if visualize:
            fig = px.scatter(x=emb[:, 0], y=emb[:, 1], color=ld_cluster_ids)
            fig.show()

        ###############################################
        # 2. 1. ABW: Ratio between avg. intra-cluster
        # distance and avg. inter-cluster distance.
        ###############################################

        rec_coords: pd.DataFrame = pd.DataFrame(emb, columns=["x", "y"])
        rec_coords["cluster_id"] = ld_cluster_ids
        inter_cluster_dists: List[np.ndarray] = []
        intra_cluster_dists: List[np.ndarray] = []

        # Compute avg. distance between points in same cluster and avg. dist for points in different clusters.
        ld_cluster_ids: list = list(rec_coords.cluster_id.unique())
        for ix1, cluster_id_1 in enumerate(ld_cluster_ids):
            # -1 represents no cluster membership, hence this should be ignored.
            if cluster_id_1 == "-1":
                continue

            # Compute avg. distance between points in same cluster.
            coords_in_cluster1: np.ndarray = rec_coords[rec_coords.cluster_id == cluster_id_1][["x", "y"]].values
            intra_cluster_dists.append(cdist(coords_in_cluster1, coords_in_cluster1, "euclidean").flatten())

            # Compute avg. distance between points in different clusters.
            for ix2, cluster_id_2 in enumerate(ld_cluster_ids[ix1 + 1:]):
                if cluster_id_2 == "-1":
                    continue
                inter_cluster_dists.append(
                    cdist(
                        coords_in_cluster1,
                        rec_coords[rec_coords.cluster_id == cluster_id_2][["x", "y"]].values,
                        "euclidean"
                    ).flatten()
                )

        # Use a default value for ABW if there's only one cluster since ABW is undefined for one cluster.
        abw: float = (
            # Normalize values based the number of clusters.
            np.concatenate(intra_cluster_dists).mean() / np.concatenate(inter_cluster_dists).mean()
            if len(rec_coords.cluster_id.unique()) != 1 else 1
        )

        ###############################################
        # 2. 2. Number of clusters.
        ###############################################

        number_of_clusters: int = len(rec_coords.cluster_id.unique())

        ###############################################
        # 2. 3. Cluster purity -> HD-ABW.
        ###############################################

        hd_dists: np.ndarray = data[ds_name]["hd_dataset"].compute_distance_matrix()
        assert emb.shape[0] == hd_dists.shape[0]

        inter_cluster_dists = []
        intra_cluster_dists = []

        # Compute avg. HD distance between points in same cluster and avg. dist for points in different clusters.
        ld_cluster_ids: list = list(rec_coords.cluster_id.unique())
        for ix1, cluster_id_1 in enumerate(ld_cluster_ids):
            # -1 represents no cluster membership, hence this should be ignored.
            if cluster_id_1 == "-1":
                continue

            # Compute avg. distance between points in same cluster.
            records_in_cluster1_ids: np.ndarray = rec_coords[rec_coords.cluster_id == cluster_id_1].index.values
            intra_cluster_dists.append(hd_dists[np.ix_(records_in_cluster1_ids, records_in_cluster1_ids)].flatten())

            # Compute avg. distance between points in different clusters.
            for ix2, cluster_id_2 in enumerate(ld_cluster_ids[ix1 + 1:]):
                if cluster_id_2 == "-1":
                    continue

                records_in_cluster2_ids: np.ndarray = rec_coords[rec_coords.cluster_id == cluster_id_2].index.values
                inter_cluster_dists.append(hd_dists[np.ix_(records_in_cluster1_ids, records_in_cluster2_ids)].flatten())

        abw_hd: float = (
            # Normalize values based the number of clusters.
            np.concatenate(intra_cluster_dists).mean() / np.concatenate(inter_cluster_dists).mean()
            if len(rec_coords.cluster_id.unique()) != 1 else 1
        )

        ###############################################
        # 2. 4. Axis/attribute alignment.
        ###############################################

        hd_data_with_emb_coords: pd.DataFrame = data[ds_name]["hd_dataset"].features().copy(
            deep=True
        ).select_dtypes(include=np.number)
        hd_data_with_emb_coords[data[ds_name]["hd_dataset"].labels().name] = data[ds_name]["hd_dataset"].labels().values
        hd_data_with_emb_coords["x"] = emb[:, 0]
        hd_data_with_emb_coords["y"] = emb[:, 1]

        # Compute correlation to
        attribute_axis_corrs: pd.DataFrame = hd_data_with_emb_coords.corr(method="spearman")
        attribute_axis_corrs: dict = {
            axis: [
                abs(attribute_axis_corrs.loc[attr][axis])
                for attr in attribute_axis_corrs if attr not in ("x", "y")
            ]
            for axis in ("x", "y")
        }
        attribute_axis_corrs_measures: dict = {}
        for axis in ("x", "y"):
            attribute_axis_corrs_measures.update({
                "axis_corr_min_" + axis: np.nan_to_num(np.min(attribute_axis_corrs[axis])),
                "axis_corr_max_" + axis: np.nan_to_num(np.max(attribute_axis_corrs[axis])),
                "axis_corr_mean_" + axis: np.nan_to_num(np.mean(attribute_axis_corrs[axis])),
                "axis_corr_median_" + axis: np.nan_to_num(np.median(attribute_axis_corrs[axis])),
                "axis_corr_sum_" + axis: np.nan_to_num(np.sum(attribute_axis_corrs[axis]))
            })

        ###############################################
        # 2. 5. Outlier percentage.
        ###############################################

        outlier_perc: float = (OneClassSVM(gamma='auto').fit_predict(emb) == 1).sum() / len(emb)

        ###############################################
        # 2. 6. Point density.
        ###############################################

        binned_emb_data, _, _ = np.histogram2d(emb[:, 0], emb[:, 1], bins=(10, 10))

        density_measures: dict = {
            "density_min": np.min(binned_emb_data),
            "density_max": np.max(binned_emb_data),
            "density_mean": binned_emb_data.mean(),
            "density_median": np.median(binned_emb_data),
            "density_std": binned_emb_data.std()
        }

        ###############################################
        # 2. 7. Unsupervised DSC (distance consistency).
        ###############################################

        # Compute clusters in HD space.
        hd_dists: np.ndarray = data[ds_name]["hd_dataset"].compute_distance_matrix()
        rec_coords: pd.DataFrame = pd.DataFrame(emb, columns=["x", "y"])
        rec_coords["hd_cluster_id"] = compute_clusters(hd_dists, metric="precomputed", min_samples=2)

        # Compute centroids each cluster in LD space.
        cluster_centroids: pd.DataFrame = rec_coords.groupby("hd_cluster_id").mean()[["x", "y"]]
        hd_cluster_centroids_ids: list = [cid for cid in cluster_centroids.index.values if cid != "-1"]
        cluster_centroids = cluster_centroids.loc[hd_cluster_centroids_ids]

        # Compute distances between records and clusters, select nearest cluster.
        rec_coords["nearest_hd_cluster_id_in_ld_space"] = np.argmin(
            cdist(rec_coords[["x", "y"]].values, cluster_centroids.values, "euclidean"),
            axis=1
        ).astype(str)

        unsupervised_dsc: float = len(rec_coords[
            (rec_coords.hd_cluster_id != "-1") &
            (rec_coords.hd_cluster_id != rec_coords.nearest_hd_cluster_id_in_ld_space)
        ]) / len(
            rec_coords[rec_coords.hd_cluster_id != "-1"]
        ) if len(rec_coords[rec_coords.hd_cluster_id != "-1"]) else 1

        ###############################################
        # 2. 8. Hypothesis margin (optional).
        ###############################################

        distance_deltas: List[float] = []
        for cluster_id in hd_cluster_centroids_ids:
            # Compute distance to closest neighbour in same cluster.
            coords_in_cluster: np.ndarray = rec_coords[rec_coords.hd_cluster_id == cluster_id][["x", "y"]].values
            coords_not_in_cluster: np.ndarray = rec_coords[
                ~rec_coords.hd_cluster_id.isin({cluster_id, "-1"})
            ][["x", "y"]].values
            intra_cluster_dists: np.ndarray = cdist(coords_in_cluster, coords_in_cluster, "euclidean")
            inter_cluster_dists: np.ndarray = cdist(coords_in_cluster, coords_not_in_cluster, "euclidean")

            # Ignore diagonale in intra_cluster_dists, since those distances will always be zero and are not interesting
            # for our HM calculation.
            np.fill_diagonal(intra_cluster_dists, np.inf)
            # Add differences between next points in the same and next points in a different cluster.
            distance_deltas.extend((np.min(intra_cluster_dists, axis=1) - np.min(inter_cluster_dists, axis=1)).flatten())

        unsupervised_hm: float = np.asarray(distance_deltas).mean()

        ###############################################
        # 2. 9. Append features.
        ###############################################

        features.append({
            "abw": abw,
            "number_of_clusters": number_of_clusters,
            "abw_hd": abw_hd,
            **attribute_axis_corrs_measures,
            "outlier_percentage": outlier_perc,
            **density_measures,
            "unsupervised_dsc": unsupervised_dsc,
            "unsupervised_hm": unsupervised_hm
        })

        pbar.update(1)
    pbar.close()

    return pd.concat(
        [
            user_scores[[
                "r_nx", "b_nx", "stress", "target_domain_performance", "separability_metric", "rating", "n_components"
            ]].reset_index(),
            pd.DataFrame(features)
        ],
        axis=1
    ).set_index(["dataset", "id"])


def train(data: pd.DataFrame, cols_to_keep: list = None):
    """
    Trains model(s) on feature data to predict user ratings.
    Currently used models:
        - Linear regression.
        - LightGBM
    :param data: Dataframe to predict.
    :param use_only_preservation_metrics: Indicates whether only preservation metrics should be used as features.
    """

    target: str = "rating"
    data = data.drop(columns="separability_metric")

    if cols_to_keep:
        data = data[[*cols_to_keep, "rating"]]
    data = data.drop(columns=["r_nx", "b_nx", "stress"])
    metrics: List[Dict] = []

    # 1. With linear regression.
    print("=== Linear regression ===")
    n_splits: int = 100
    pbar: tqdm = tqdm(total=n_splits)
    for train_indices, test_indices in ShuffleSplit(n_splits=n_splits, test_size=.2).split(data):
        features: np.ndarray = data.drop(columns=target).values

        # Split in train and test set.
        train_feats: np.ndarray = features[train_indices, :]
        train_labels: np.ndarray = data[[target]].values[train_indices, :]
        test_feats: np.ndarray = features[test_indices, :]
        test_labels: np.ndarray = data[[target]].values[test_indices, :]

        # Normalize features.
        scaler: StandardScaler = StandardScaler()
        scaler.fit(train_feats)

        estimator = linear_model.Lasso(alpha=0.015, max_iter=2000)

        # Train model.
        estimator.fit(train_feats, train_labels)
        test_labels_predicted: np.ndarray = estimator.predict(test_feats)
        # print('Coefficients: \n', estimator.coef_)

        metrics.append({
            "model": "linear_regression",
            "mean_squared_error": mean_squared_error(test_labels, test_labels_predicted),
            "mean_absolute_error": mean_absolute_error(test_labels, test_labels_predicted),
            "median_absolute_error": median_absolute_error(test_labels, test_labels_predicted),
            "explained_variance": explained_variance_score(test_labels, test_labels_predicted)
        })

        pbar.update(1)
    pbar.close()

    # 2. With boosting (LightGBM).
    n_splits: int = 100
    print("=== Boosting ===")
    pbar: tqdm = tqdm(total=n_splits)
    best_params: dict = dict()
    tuning_history: list = list()
    model: Optional[optuna_lgbm.Booster] = None
    test_feats: Optional[np.ndarray] = None
    test_labels: Optional[np.ndarray] = None
    cols: list = data.drop(columns=target).columns

    for train_indices, test_indices in ShuffleSplit(n_splits=n_splits, test_size=.2).split(data):
        features: np.ndarray = data.drop(columns=target).values

        # Split in train and test set.
        train_feats: np.ndarray = features[train_indices, :]
        train_labels: np.ndarray = data[[target]].values[train_indices, :]
        test_feats: np.ndarray = features[test_indices, :]
        test_labels: np.ndarray = data[[target]].values[test_indices, :]

        if True or not len(best_params):
            # Split train set in train and validation set.
            train_feats, val_feats, train_labels, val_labels = train_test_split(
                train_feats, train_labels, test_size=0.2
            )

            dtrain: lgbm.Dataset = lgbm.Dataset(
                pd.DataFrame(train_feats, columns=cols),
                label=train_labels.ravel().tolist(),
                params={'verbose': -1}
            )
            dval: lgbm.Dataset = lgbm.Dataset(
                pd.DataFrame(val_feats, columns=cols),
                label=val_labels.ravel().tolist(),
                params={'verbose': -1}
            )

            params: dict = {
                "objective": "regression",
                "metric": "l2",
                "verbosity": -1,
                "verbose": -1,
                "silent": True,
                "boosting_type": "gbdt",
            }

            model: optuna_lgbm.Booster = optuna_lgbm.train(
                params,
                dtrain,
                valid_sets=[dtrain, dval],
                early_stopping_rounds=100,
                verbosity=-1,
                verbose_eval=False,
                best_params=best_params,
                tuning_history=tuning_history
            )
            test_labels_predicted: np.ndarray = model.predict(
                pd.DataFrame(test_feats, columns=cols), num_iteration=model.best_iteration
            )
        else:
            model: lgbm.LGBMRegressor = lgbm.LGBMRegressor(**best_params)
            model.fit(train_feats, train_labels)
            test_labels_predicted: np.ndarray = model.predict(test_feats)

        metrics.append({
            "model": "boosting",
            "mean_squared_error": mean_squared_error(test_labels, test_labels_predicted),
            "mean_absolute_error": mean_absolute_error(test_labels, test_labels_predicted),
            "median_absolute_error": median_absolute_error(test_labels, test_labels_predicted),
            "explained_variance": explained_variance_score(test_labels, test_labels_predicted)
        })

        pbar.update(1)
    pbar.close()

    """
    just preservation:
        lasso ->
            mean_squared_error       1.126817
            mean_absolute_error      0.856218
            median_absolute_error    0.751933
            explained_variance       0.427465
        boost ->
            mean_squared_error       1.117067
            mean_absolute_error      0.818996
            median_absolute_error    0.630537
            explained_variance       0.410680
    """
    """
    with all features:
        lasso ->
            mean_squared_error       1.066163
            mean_absolute_error      0.831639
            median_absolute_error    0.689467
            explained_variance       0.440250
        boost ->
            an_squared_error       1.178090
            mean_absolute_error      0.848246
            median_absolute_error    0.671124
            explained_variance       0.398336
    """

    metrics: pd.DataFrame = pd.DataFrame(metrics)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(metrics[metrics.model == "linear_regression"].mean())
        print(metrics[metrics.model == "boosting"].mean())

    ###########################
    # Interpret models.
    ###########################

    # https://www.kaggle.com/slundberg/interpreting-a-lightgbm-model
    shap_values: np.ndarray = shap.TreeExplainer(model).shap_values(pd.DataFrame(test_feats, columns=cols))
    global_importances: np.ndarray = np.abs(shap_values).mean(0)#[:-1]

    inds = np.argsort(-global_importances)
    f = plt.figure(figsize=(5, 10))
    y_pos = np.arange(min(20, test_feats.shape[1]))
    inds2 = np.flip(inds[:20], 0)
    plt.barh(y_pos, global_importances[inds2], align='center', color="#1E88E5")
    plt.yticks(y_pos, fontsize=13)
    plt.gca().set_yticklabels(data.columns[inds2])
    plt.xlabel('mean abs. SHAP value (impact on model output)', fontsize=13)
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.grid()
    plt.show()

    shap.summary_plot(shap_values, pd.DataFrame(test_feats, columns=cols))
    plt.show()

    
if __name__ == '__main__':
    base_data_path: str = "/home/raphael/Development/data/TALE/"
    force_recomputation: bool = False

    # Engineer (visual quality criterion) features
    cached_file_path: str = "/tmp/tale_feature_data.pkl"
    if force_recomputation or not os.path.isfile(cached_file_path):
        # Build dataset with all necessary data for those datasets we have been collecting data for.
        # Note that
        #   (1) we distinguish between datasets but not between DR algorithms, since we do not consider the latter
        #       explicitly during analysis but the former has to be provided since we incorporate HD data properties
        #       into some features and targets.
        #   (2) We only used t-SNE for our evaluation, so we only need to consider our t-SNE embeddings' properties.
        datasets: dict = {
            ds_name: {
                "hd_dataset": generate_instance(ds_name, base_data_path + ds_name),
                "embedding_data": open_file(filename=base_data_path + ds_name + "/embedding_tsne.h5", mode="r+"),
            }
            for ds_name in ["happiness", "movie"]
        }

        engineer_features(
            data=datasets,
            user_scores=read_user_scores(
                base_data_path + "TALE-study"
            ).reset_index().set_index(["dataset", "id"]).drop(columns=["index"])
        ).to_pickle(cached_file_path)

    # Load engineered feature set.
    feature_data: pd.DataFrame = pd.read_pickle(cached_file_path)

    # Train model.
    train(data=feature_data)
