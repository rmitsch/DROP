"""
Analysis of perceptual embedding quality.
Includes engineering of (visual quality criterion) features and building a model to predict perceived embedding quality.
"""
import math
from pprint import pprint
from typing import Dict, List, Tuple, Optional

import hdbscan
from scipy import stats
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


def compute_clusters(embedding: np.ndarray) -> Tuple[str]:
    """
    Compute clusters in an embedding.
    :param embedding: Embedding coordinates.
    :return: Tuple of cluster IDs, one per record in embedding. Note that -1 indicates no cluster association, records
    should hence not be treated as cluster.
    """

    return tuple([
        str(label) for label in hdbscan.HDBSCAN(
            metric="euclidean",
            min_samples=15
        ).fit(embedding).labels_
    ])


def analyse(data: Dict, user_scores: pd.DataFrame):
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
        cluster_ids: Tuple = compute_clusters(embeddings[ds_name][emb_id])
        emb: np.ndarray = embeddings[ds_name][emb_id]
        # Extend by second dimension, if only 1D.
        if emb.shape[1] == 1:
            emb = np.asarray([emb[:, 0], [0] * len(emb)]).T

        if visualize:
            fig = px.scatter(x=emb[:, 0], y=emb[:, 1], color=cluster_ids)
            fig.show()

        ###############################################
        # 2. 1. ABW: Ratio between avg. intra-cluster
        # distance and avg. inter-cluster distance.
        ###############################################

        rec_coords: pd.DataFrame = pd.DataFrame(emb, columns=["x", "y"])
        rec_coords["cluster_id"] = cluster_ids
        inter_cluster_dists: List[np.ndarray] = []
        intra_cluster_dists: List[np.ndarray] = []

        # Compute avg. distance between points in same cluster and avg. dist for points in different clusters.
        cluster_ids: list = list(rec_coords.cluster_id.unique())
        for ix1, cluster_id_1 in enumerate(cluster_ids):
            # -1 represents no cluster membership, hence this should be ignored.
            if cluster_id_1 == "-1":
                continue

            # Compute avg. distance between points in same cluster.
            coords_in_cluster1: np.ndarray = rec_coords[rec_coords.cluster_id == cluster_id_1][["x", "y"]].values
            intra_cluster_dists.append(cdist(coords_in_cluster1, coords_in_cluster1, "euclidean").flatten())

            # Compute avg. distance between points in different clusters.
            for ix2, cluster_id_2 in enumerate(cluster_ids[ix1 + 1:]):
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
        cluster_ids: list = list(rec_coords.cluster_id.unique())
        for ix1, cluster_id_1 in enumerate(cluster_ids):
            # -1 represents no cluster membership, hence this should be ignored.
            if cluster_id_1 == "-1":
                continue

            # Compute avg. distance between points in same cluster.
            records_in_cluster1_ids: np.ndarray = rec_coords[rec_coords.cluster_id == cluster_id_1].index.values
            intra_cluster_dists.append(hd_dists[np.ix_(records_in_cluster1_ids, records_in_cluster1_ids)].flatten())

            # Compute avg. distance between points in different clusters.
            for ix2, cluster_id_2 in enumerate(cluster_ids[ix1 + 1:]):
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

        ###############################################
        # 2. 8. Hypothesis margin (optional).
        ###############################################

        ###############################################
        # 2. x. Append features.
        ###############################################

        features.append({
            "abw": abw,
            "number_of_clusters": number_of_clusters,
            "abw_hd": abw_hd,
            **attribute_axis_corrs_measures,
            "outlier_percentage": outlier_perc,
            **density_measures
        })

        pbar.update(1)
    pbar.close()

    data: pd.DataFrame = pd.concat(
        [
            user_scores[[
                "r_nx", "b_nx", "stress", "target_domain_performance", "separability_metric", "rating"
            ]].reset_index(),
            pd.DataFrame(features)
        ],
        axis=1
    ).set_index(["dataset", "id"])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(data.head(20))
        print(len(data))

    # todo - next steps:
    #  - unsupervised DSC
    #  - hypothesis margin (?)
    #  - feature selection
    #  - model training and optimization
    #  - feature importance analysis
    #     - use explainer/surrogate models - SHAP, skope-rules?


if __name__ == '__main__':
    base_data_path: str = "/home/raphael/Development/data/TALE/"

    # Build dataset with all necessary data for those datasets we have been collecting data for.
    # Note that
    #   (1) we distinguish between datasets but not between DR algorithms, since we do not consider the latter
    #       explicitly during analysis but the former has to be provided since we incorporate HD data properties into
    #       some features and targets.
    #   (2) We only used t-SNE for our evaluation, so we only need to consider our t-SNE embeddings' properties.
    datasets: dict = {
        ds_name: {
            "hd_dataset": generate_instance(ds_name, base_data_path + ds_name),
            "embedding_data": open_file(filename=base_data_path + ds_name + "/embedding_tsne.h5", mode="r+"),
        }
        for ds_name in ["happiness", "movie"]

    }

    # Analyse embedding data.
    analyse(
        data=datasets,
        user_scores=read_user_scores(
            base_data_path + "TALE-study"
        ).reset_index().set_index(["dataset", "id"]).drop(columns=["index"])
    )

