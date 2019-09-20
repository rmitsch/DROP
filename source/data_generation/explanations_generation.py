import os
import pickle

import psutil
from tqdm import tqdm
import tables
import pandas as pd
import shap
import logging
from multiprocessing import Pool
from functools import partial
from utils import Utils
from data_generation.dimensionality_reduction import DimensionalityReductionKernel


def construct_explanations(
        embedding_record: tuple,
        original_df: pd.DataFrame,
        features_preprocessed: pd.DataFrame,
        surrogate_models: dict,
        metadata_template: dict
) -> list:
    """
    Construct local explanations as SHAP values.
    :param embedding_record:
    :param original_df:
    :param features_preprocessed:
    :param surrogate_models:
    :param metadata_template:
    :return: List of tupels with schema (record ID, HP, O, correlation strength).
    """

    active_col_indices: list = Utils.get_active_col_indices(
        features_preprocessed, {"original": original_df}, int(embedding_record[0])
    )
    max_per_objective: dict = {obj: original_df[obj].max() for obj in metadata_template["objectives"]}
    cols: list = features_preprocessed.columns.values[active_col_indices]

    # Pack results of SHAP values for hyperparameters together with record ID and correspondig objectives.
    # Flatten to 1D list before returning it.
    return [
        item for sublist
        in [
            [
                *zip(
                    [int(embedding_record[0])] * len(cols),
                    [objective] * len(cols),
                    cols,
                    # See https://github.com/slundberg/shap/issues/392 on how to verify predicted SHAP values.
                    shap.TreeExplainer(
                        surrogate_models[objective]
                    ).shap_values(
                        embedding_record[1].values, approximate=False
                        # Transform SHAP values of objectives w/o upper bounds into [0, 1]-interval by dividing values
                        # for unbounded objectives  through the maximum for this objective.
                        # Note that we assume all objectives, including those w/o upper bounds, to be [0, x] where x is
                        # either 1 or an arbitrary real number.
                        # Hence we iterate over upper-unbounded objectives, get their max, divide values in explanations
                        # through the maximum of that objective. This yields [0, 1]-intervals for all objectives.
                    )[active_col_indices].tolist() / max_per_objective[objective]
                )
            ]
            for objective in metadata_template["objectives"]
        ]
        for item in sublist
    ]


def compute_and_persist_explainer_values(
        logger: logging.Logger, storage_path: str, dr_kernel_name: str, dataset_name: str
):

    logger.info("  Fetching data.")
    # 1. Get metadata template.
    metadata_template = Utils.get_metadata_template(
        DimensionalityReductionKernel.DIM_RED_KERNELS[dr_kernel_name]
    )

    # 2. Load data.
    h5file = tables.open_file(
        filename=storage_path + "drop_" + dataset_name + "_" + dr_kernel_name.lower() + ".h5",
        mode="r+"
    )
    df = pd.DataFrame(h5file.root.metadata[:]).set_index("id")
    h5file.close()
    features_preprocessed, labels, _ = Utils.preprocess_embedding_metadata_for_predictor(
        metadata_template=metadata_template, embeddings_metadata=df
    )

    # 3. Compute and store surrogate models.
    logger.info("  Training surrogate models.")
    surrogate_models: dict = Utils.fit_random_forest_regressors(
        metadata_template=metadata_template,
        features_df=features_preprocessed,
        labels_df=labels
    )
    with open(storage_path + dataset_name.lower() + "_" + dr_kernel_name.lower() + "_surrogatemodels.pkl", "wb") as file:
        pickle.dump(surrogate_models, file)

    # 4. Compute and store SHAP values.
    logger.info("  Computing SHAP values.")
    records: list = [(ix, record) for ix, record in features_preprocessed.iterrows()]
    with Pool(psutil.cpu_count(logical=False)) as pool:
        results: list = list(
            tqdm(
                pool.imap(
                    partial(
                        construct_explanations,
                        original_df=df,
                        features_preprocessed=features_preprocessed,
                        surrogate_models=surrogate_models,
                        metadata_template=metadata_template
                    ),
                    records
                ),
                total=len(features_preprocessed)
            )
        )

        results: pd.DataFrame = pd.DataFrame(
            [item for sublist in results for item in sublist],
            columns=["id", "objective", "hyperparameter", "value"]
        ).set_index("id")

    results.to_pickle(storage_path + dataset_name.lower() + "_" + dr_kernel_name.lower() + "_explainervalues.pkl")


if __name__ == '__main__':
    compute_and_persist_explainer_values(
        Utils.create_logger(), os.getcwd() + "/data/", "TSNE", "happiness"
    )
