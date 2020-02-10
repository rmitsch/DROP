import os
import pickle

from tables import File
from tqdm import tqdm
import tables
import pandas as pd
import shap
import logging
import math
from utils import Utils
from data_generation.dimensionality_reduction import DimensionalityReductionKernel
import numpy as np


def construct_explanations(
        features_preprocessed: pd.DataFrame,
        original_df: pd.DataFrame,
        surrogate_models: dict,
        objectives: list
) -> pd.DataFrame:
    """
    Construct local explanations as SHAP values.
    :param features_preprocessed: Set of preprocessed features.
    :param original_df: Original (i.e. w/o explanations) dataframe.
    :param surrogate_models: Surrogate models per objective.
    :param objectives: List of objectives.
    :return:
    """
    records: list = [(ix, record) for ix, record in features_preprocessed.iterrows()]

    # Initialize SHAP explainers (one per objective).
    explainers: dict = {
        objective: shap.TreeExplainer(surrogate_models[objective])
        for objective in objectives
    }
    # Select indices of active columns (i.e. ignore columns irrelevant for explanations).
    active_col_indices: list = Utils.get_active_col_indices(
        features_preprocessed, {"original": original_df}, int(records[0][0])
    )
    active_cols: list = features_preprocessed.columns[active_col_indices]
    # Compute maxima per objecive.
    max_per_objective: dict = {obj: original_df[obj].max() for obj in explainers}

    n_chunks: int = 100
    chunksize: int = math.ceil(len(records) / n_chunks)
    pbar: tqdm = tqdm(total=n_chunks)
    explanations: list = []
    for i in range(n_chunks):
        curr_records: list = records[i * chunksize:(i + 1) * chunksize]

        for obj in objectives:
            df: pd.DataFrame = pd.DataFrame((
                explainers[obj].shap_values(
                    # Transform SHAP values of objectives w/o upper bounds into [0, 1]-interval by dividing values
                    # for unbounded objectives through the maximum for this objective.
                    # Note that we assume all objectives, including those w/o upper bounds, to be [0, x] where x is
                    # either 1 or an arbitrary real number.
                    # Hence we iterate over upper-unbounded objectives, get their max, divide values in explanations
                    # through the maximum of that objective. This yields [0, 1]-intervals for all objectives.
                    np.asarray([record[1].values for record in curr_records]), approximate=False
                )[:, active_col_indices].tolist() / max_per_objective[obj]
            ), columns=active_cols)
            df["id"] = [int(record[0]) for record in curr_records]
            df["objective"] = obj
            explanations.append(
                df.melt(id_vars=["id", "objective"], value_vars=active_cols, var_name="hyperparameter").set_index("id")
            )

        pbar.update(1)
    pbar.close()

    return pd.concat(explanations)


def compute_and_persist_explainer_values(
        logger: logging.Logger, storage_path: str, dr_kernel_name: str, dataset_name: str
):
    logger.info("  Fetching data.")

    ######################################
    # 1. Get metadata template.
    ######################################

    metadata_template: dict = Utils.get_metadata_template(
        DimensionalityReductionKernel.DIM_RED_KERNELS[dr_kernel_name]
    )

    ######################################
    # 2. Load data.
    ######################################

    h5file: File = tables.open_file(
        filename=storage_path + "/tale_" + dataset_name + "_" + dr_kernel_name.lower() + ".h5",
        mode="r+"
    )
    df = pd.DataFrame(h5file.root.metadata[:]).set_index("id")
    h5file.close()
    features_preprocessed, labels, _ = Utils.preprocess_embedding_metadata_for_predictor(
        metadata_template=metadata_template, embeddings_metadata=df
    )

    ######################################
    # 3. Compute and store surrogate models.
    ######################################

    logger.info("  Training surrogate models.")

    surrogate_models: dict = Utils.fit_surrogate_regressor(
        metadata_template=metadata_template, features_df=features_preprocessed, labels_df=labels
    )
    with open(
            storage_path + "/" + dataset_name.lower() + "_" + dr_kernel_name.lower() + "_surrogatemodels.pkl", "wb"
    ) as file:
        pickle.dump(surrogate_models, file)

    ######################################
    # 4. Compute and store SHAP values.
    ######################################

    logger.info("  Computing SHAP values.")
    construct_explanations(
        features_preprocessed, df, surrogate_models, metadata_template["objectives"]
    ).to_pickle(
        storage_path + "/" + dataset_name.lower() + "_" + dr_kernel_name.lower() + "_explainervalues.pkl"
    )


if __name__ == '__main__':
    compute_and_persist_explainer_values(
        Utils.create_logger(), os.getcwd() + "/data/", "TSNE", "happiness"
    )
