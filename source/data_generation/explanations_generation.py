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
from data_generation.dimensionality_reduction.DimensionalityReductionKernelParameters import DIM_RED_KERNELS_PARAMETERS
import numpy as np


def construct_local_explanations(
        features: pd.DataFrame,
        labels: pd.DataFrame,
        original_df: pd.DataFrame,
        surrogate_models: dict,
        objectives: list,
        dim_red_kernel_name: str
) -> pd.DataFrame:
    """
    Construct local explanations as SHAP values.
    :param features: Set of preprocessed features.
    :param labels: Labels for dataset.
    :param original_df: Original (i.e. w/o explanations) dataframe.
    :param surrogate_models: Surrogate models per objective.
    :param objectives: List of objectives.
    :param dim_red_kernel_name:
    :return: Dataframe with local explanations.
    """

    feature_values: np.ndarray = features.values
    record_ids: np.ndarray = features.index.values

    ########################################################
    # Initialize SHAP explainers (one per objective).
    ########################################################

    explainers: dict = {
        objective: shap.TreeExplainer(surrogate_models[objective])
        for objective in objectives
    }

    ########################################################
    # Compose set of active columns' indices.
    ########################################################

    active_columns: pd.DataFrame = Utils.gather_column_activity(
        features, DIM_RED_KERNELS_PARAMETERS[dim_red_kernel_name]
    )

    ########################################################
    # Compute SHAP explanations.
    ########################################################

    # Compute maxima per objecive.
    max_per_objective: dict = {obj: original_df[obj].max() for obj in explainers}

    chunksize: int = 10
    n_chunks: int = math.ceil(len(feature_values) / chunksize)
    pbar: tqdm = tqdm(total=n_chunks)
    explanations: list = []
    for i in range(n_chunks):
        curr_records: np.ndarray = feature_values[i * chunksize:(i + 1) * chunksize, :]
        curr_record_ids: np.ndarray = record_ids[i * chunksize:(i + 1) * chunksize].astype(int)

        for obj in objectives:
            df: pd.DataFrame = pd.DataFrame(
                explainers[obj].shap_values(
                    # Transform SHAP values of objectives w/o upper bounds into [0, 1]-interval by dividing values
                    # for unbounded objectives through the maximum for this objective.
                    # Note that we assume all objectives, including those w/o upper bounds, to be [0, x] where x is
                    # either 1 or an arbitrary real number.
                    # Hence we iterate over upper-unbounded objectives, get their max, divide values in explanations
                    # through the maximum of that objective. This yields [0, 1]-intervals for all objectives.
                    np.asarray([record for record in curr_records]),
                    approximate=False
                )[
                    # Select only active columns. Use fancy indexing to select active columns for that - see
                    # https://stackoverflow.com/questions/20103779/index-2d-numpy-array-by-a-2d-array-of-indices-without-loops.
                    np.arange(len(curr_records))[:, None],
                    np.asarray(active_columns.loc[curr_record_ids].idx.values.tolist())
                ].tolist() / max_per_objective[obj]
            )

            # Complement with metadata; join active_columns to get real hyperparameter name.
            df["id"] = curr_record_ids
            df["objective"] = obj
            df = df.melt(id_vars=["id", "objective"], var_name="hyperparameter").set_index("id").join(active_columns)
            df.hyperparameter = df.apply(lambda row: features.columns[row["idx"][row["hyperparameter"]]], axis=1)
            # Add to collection.
            explanations.append(df.drop(columns=["idx", "cols"]))
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
        filename=storage_path + "/embedding_" + dr_kernel_name.lower() + ".h5",
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
            storage_path + "/" + dr_kernel_name.lower() + "_surrogatemodels.pkl", "wb"
    ) as file:
        pickle.dump(surrogate_models, file)

    ######################################
    # 4. Compute and store SHAP values.
    ######################################

    logger.info("  Computing SHAP values.")
    construct_local_explanations(
        features_preprocessed, labels, df, surrogate_models, metadata_template["objectives"], dr_kernel_name
    ).to_pickle(
        storage_path + "/" + dr_kernel_name.lower() + "_explainervalues.pkl"
    )


if __name__ == '__main__':
    compute_and_persist_explainer_values(
        Utils.create_logger(), os.getcwd() + "/data/", "TSNE", "happiness"
    )
