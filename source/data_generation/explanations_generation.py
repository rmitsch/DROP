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
) -> dict:
    """
    Construct local explanations as SHAP values.
    :param embedding_record:
    :param original_df:
    :param features_preprocessed:
    :param surrogate_models:
    :param metadata_template:
    :return:
    """

    active_col_indices: list = Utils.get_active_col_indices(
        features_preprocessed, {"original": original_df}, int(embedding_record[0])
    )

    return {
        "idx": int(embedding_record[0]),
        "explainer_models": {
            objective: shap.TreeExplainer(
                surrogate_models[objective]
            ).shap_values(embedding_record[1].values, approximate=False)[active_col_indices].tolist()
            for objective in metadata_template["objectives"]
        }
    }


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
    logger.info("  Computing surrogate models.")
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
        results: dict = {k: v for d in results for k, v in d.items()}
    with open(storage_path + dataset_name.lower() + "_" + dr_kernel_name.lower() + "_explainervalues.pkl", "wb") as file:
        pickle.dump(results, file)


if __name__ == '__main__':
    compute_and_persist_explainer_values(
        Utils.create_logger(), os.getcwd() + "/data/", "TSNE", "happiness"
    )
