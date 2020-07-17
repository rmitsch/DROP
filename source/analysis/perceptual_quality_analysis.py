"""
Analysis of perceptual embedding quality.
Includes engineering of (visual quality criterion) features and building a model to predict perceived embedding quality.
"""
import math
from typing import Dict, List, Tuple, Optional

import hdbscan
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
    visualize: bool = True

    for ix, row in training_data.iterrows():
        ds_name, emb_id = ix
        cluster_ids: Tuple = compute_clusters(embeddings[ds_name][emb_id])
        emb: np.ndarray = embeddings[ds_name][emb_id]
        if emb.shape[1] == 1:
            emb = np.asarray([emb[:, 0], [0] * len(emb)]).T

        if visualize:
            fig = px.scatter(x=emb[:, 0], y=emb[:, 1], color=cluster_ids)
            fig.show()

        ###############################################
        # 2. 1. ABW: Ratio between avg. intra-cluster
        # distance and avg. inter-cluster distance.
        ###############################################

        record_coordinates: pd.DataFrame = pd.DataFrame(emb, columns=["x", "y"])
        record_coordinates["cluster_id"] = cluster_ids
        avg_inter_cluster_dist: float = 0
        avg_intra_cluster_dist: float = 0

        # Compute avg. distance between points in same cluster and avg. dist for points in different clusters.
        cluster_ids: list = list(record_coordinates.cluster_id.unique())
        for ix1, cluster_id_1 in enumerate(cluster_ids):
            # -1 represents no cluster membership, hence this should be ignored.
            if cluster_id_1 == "-1":
                continue

            # Compute avg. distance between points in same cluster.
            coords_in_cluster1: np.ndarray = record_coordinates[
                record_coordinates.cluster_id == cluster_id_1
            ][["x", "y"]].values
            avg_intra_cluster_dist += cdist(coords_in_cluster1, coords_in_cluster1, "euclidean").mean()

            # Compute avg. distance between points in different clusters.
            for ix2, cluster_id_2 in enumerate(cluster_ids[ix1 + 1:]):
                if cluster_id_2 == "-1":
                    continue
                avg_inter_cluster_dist += cdist(
                    coords_in_cluster1,
                    record_coordinates[record_coordinates.cluster_id == cluster_id_2][["x", "y"]].values,
                    "euclidean"
                ).mean()

        # Use a default value for ABW if there's only one cluster since ABW is undefined for one cluster.
        abw: float = (
            # Normalize values based the number of clusters.
            (avg_intra_cluster_dist / len(cluster_ids) - 1) /
            (avg_inter_cluster_dist / (len(cluster_ids) - 1) * (len(cluster_ids) - 2) / 2)
            if len(record_coordinates.cluster_id.unique()) != 1 else 1
        )

        print("abw =", abw, avg_intra_cluster_dist, avg_inter_cluster_dist)


        ###############################################
        # 2. x. Append features.
        ###############################################

        features.append({"abw": abw})

        input()

        # Detect clusters.

        pbar.update(1)
    pbar.close()


if __name__ == '__main__':

    # Build dataset with all necessary data for those datasets we have been collecting data for.
    # Note that
    #   (1) we distinguish between datasets but not between DR algorithms, since we do not consider the latter
    #       explicitly during analysis but the former has to be provided since we incorporate HD data properties into
    #       some features and targets.
    #   (2) We only used t-SNE for our evaluation, so we only need to consider our t-SNE embeddings' properties.

    base_data_path: str = "/home/raphael/Development/data/TALE/"

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

