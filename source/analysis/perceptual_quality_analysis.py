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
import pickle
from pprint import pprint
from typing import Tuple, Dict

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tables import open_file
from tqdm import tqdm

from source.analysis.dr_measure_correlations_to_user_score import read_user_scores
from generate_data import generate_instance
import source.analysis.analysis_functions as af


if __name__ == '__main__':
    base_data_path: str = "/home/raphael/Development/data/TALE/"
    force_recomputation_features: bool = False
    force_recomputation_model: bool = True

    ###################################################
    # 1. Engineer (visual quality criterion) features
    ###################################################

    cached_features_file_path: str = "/tmp/tale_feature_data.pkl"
    if force_recomputation_features or not os.path.isfile(cached_features_file_path):
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

        af.engineer_features(
            data=datasets,
            user_scores=read_user_scores(
                base_data_path + "TALE-study"
            ).reset_index().set_index(["dataset", "id"]).drop(columns=["index"])
        ).to_pickle(cached_features_file_path)

    # Load engineered feature set.
    feature_data: pd.DataFrame = pd.read_pickle(cached_features_file_path).drop(columns="separability_metric")

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # import numpy as np
    #
    # f, ax = plt.subplots(figsize=(10, 8))
    # corr = feature_data.corr()
    # sns.heatmap(
    #     corr.abs(),
    #     mask=np.zeros_like(corr, dtype=np.bool),
    #     cmap=sns.light_palette("red", as_cmap=True),
    #     square=True,
    #     ax=ax
    # )
    # plt.show()

    ###################################################
    # 2. Train model.
    ###################################################

    feature_configs: Dict[str, Tuple[str]] = {
        "all": tuple(c for c in feature_data.columns if c != "rating"),
        "preservation_criteria": ("r_nx", "b_nx", "stress", "target_domain_performance"),
        "vqc": tuple(
            c for c in feature_data if c not in
            ("r_nx", "b_nx", "stress", "target_domain_performance", "rating")
        ),
        "r_nx": ("r_nx",),
        "b_nx": ("b_nx",),
        "stress": ("stress", ),
        "target_domain_performance": ("target_domain_performance",)
    }
    for col in feature_data.columns:
        if col != "rating":
            feature_configs[col] = (col,)

    pbar: tqdm = tqdm(total=len(feature_configs))
    for feature_set_to_use in feature_configs:
        cached_training_data_file_path: str = base_data_path + "evaluation/results_" + feature_set_to_use + ".pkl"

        if force_recomputation_model or not os.path.isfile(cached_training_data_file_path):
            lasso_estimator, lgbm_estimator, metrics, selected_feature_data, test_feats, test_labels = af.train(
                data=feature_data, cols_to_keep=feature_configs[feature_set_to_use], filter_by_vif=True
            )

            with open(cached_training_data_file_path, 'wb') as handle:
                pickle.dump(
                    (lasso_estimator, lgbm_estimator, metrics, selected_feature_data, test_feats, test_labels),
                    handle
                )

        with open(cached_training_data_file_path, 'rb') as handle:
            (
                lasso_estimator,
                lgbm_estimator,
                metrics,
                selected_feature_data,
                test_feats,
                test_labels
            ) = pickle.load(handle)

        ###################################################
        # 3. Interpret model.
        ###################################################

        af.interpret_model(
            selected_feature_data,
            lasso_estimator,
            lgbm_estimator,
            metrics,
            test_feats,
            test_labels,
            base_data_path + "evaluation/" + feature_set_to_use + "_"
        )

        pbar.update(1)

    pbar.close()
