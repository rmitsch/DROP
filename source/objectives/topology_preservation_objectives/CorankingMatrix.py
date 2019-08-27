import pickle

import pandas as pd
import numpy as np
from objectives.DimensionalityReductionObjective import DimensionalityReductionObjective
from coranking.metrics import trustworthiness, continuity
from typing import Tuple
from scipy.spatial import distance
import sklearn
import networkx as nx
from utils import Utils
import itertools
from scipy.spatial.distance import cdist


class CorankingMatrix:
    """
    Contains representation of coranking matrix including auxiliary functions.
    """

    def __init__(
            self,
            low_dimensional_data: np.ndarray,
            high_dimensional_data: np.ndarray,
            distance_metric: str = 'euclidean',
            high_dimensional_neighbourhood_ranking: np.ndarray = None
    ):
        """
        Computes new co-ranking matrix.
        :param low_dimensional_data:
        :param high_dimensional_data:
        :param distance_metric:
        :param high_dimensional_neighbourhood_ranking:
        """

        self._distance_metric = distance_metric
        self._high_dim_ranking = None
        self._low_dim_ranking = None

        self._matrix, self._record_bin_indices = self._generate_coranking_matrix(
            high_dimensional_data=high_dimensional_data,
            low_dimensional_data=low_dimensional_data,
            high_dim_neighbourhood_ranking=high_dimensional_neighbourhood_ranking
        )

    @property
    def record_bin_indices(self) -> pd.DataFrame:
        return self._record_bin_indices

    @staticmethod
    def generate_neighbourhood_ranking(distance_matrix: np.ndarray) -> np.ndarray:
        """
        Generates neighbourhood ranking. Used externally, accepts pre-calculated distance matrix.
        Might be used to generate a ranking once and supplying it to generate_coranking_matrix() instead of calculating
        it over and over.
        :param distance_matrix
        :return: ndarray of sorted and ranked neigbourhood similarity.
        """

        # Original approach: Re-calculate with chosen data and distance metric.
        # Calculate distances, then sort by similarities.
        return distance_matrix.argsort(axis=1).argsort(axis=1)

    @staticmethod
    def _generate_neighbourhood_matrix(data: np.ndarray, distance_metric: str) -> np.ndarray:
        """
        Generates neighbourhood matrix. Accepts original data and chosen distance metric.
        Might be used to generate a ranking once and supplying it to generate_coranking_matrix() instead of calculating
        it repeatedly.
        :param data:
        :param distance_metric:
        :return: ndarray of sorted and ranked neigbourhood similarity.
        """

        # Original approach: Re-calculate with chosen data and distance metric.
        # Calculate distances, then sort by similarities.
        return distance.squareform(distance.pdist(data, distance_metric)).argsort(axis=1).argsort(axis=1)

    def _generate_coranking_matrix(
            self,
            high_dimensional_data: np.ndarray,
            low_dimensional_data: np.ndarray,
            high_dim_neighbourhood_ranking: np.ndarray = None,
            use_geodesic: bool = False
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        This function allows to construct coranking matrix based on data in state and latent space. Based on
        implementations in https://github.com/samueljackson92/coranking and
        https://git.apere.me/apere/INRIA-Internship/src/1a73ba2619f13ec23cc860309dab7fe69fd18ab9/src/embqual.py.
        Adaption: Parameter for choice of distance metric,
        miscellaneous refactoring.

        :param high_dimensional_data:
        :param low_dimensional_data:
        :param use_geodesic: Whether to use the geodesic distance for state space.
        :param high_dim_neighbourhood_ranking: Ranking of neighbourhood similarities. Calculated if none is supplied. If
        supplied, high_dimensional_data is not used.
        :return: (1) Coranking matrix as 2-dim. ndarry, (2) pd.DataFrame with records like [source record index, neighbour
        record index, bin index x-axis, bin index y-axis] if compute_indices_bin_location == True - otherwise None.
        """

        n, m = high_dimensional_data.shape

        # ------------------------------------------------------------------------------------
        # 1. Calculate ranking in high dimensional space only if that hasn't been done yet.
        # ------------------------------------------------------------------------------------

        if high_dim_neighbourhood_ranking is None:
            # Check use_geodesic to determine whether to use geodesic or spatial distances.
            self._high_dim_ranking = \
                CorankingMatrix._calculate_geodesic_ranking(
                    data=high_dimensional_data,
                    distance_metric=self._distance_metric
                ) if use_geodesic else \
                CorankingMatrix._generate_neighbourhood_matrix(
                    high_dimensional_data,
                    distance_metric=self._distance_metric
                )

        # If high_dim_neighbourhood ranking was already supplied, use it instead of calculating anew.
        else:
            self._high_dim_ranking = high_dim_neighbourhood_ranking

        # ------------------------------------------------------------------------------------
        # 2. Calculate ranking in low-dimensional space.
        # ------------------------------------------------------------------------------------

        # Calculate distances.
        self._low_dim_ranking = CorankingMatrix._generate_neighbourhood_matrix(
            low_dimensional_data,
            distance_metric=self._distance_metric
        )

        # ------------------------------------------------------------------------------------
        # 3. Compute coranking matrix.
        # ------------------------------------------------------------------------------------

        high_dim_neighbourhood_positions = self._high_dim_ranking.flatten()
        low_dim_neighbourhood_positions = self._low_dim_ranking.flatten()

        Q, xedges, yedges = np.histogram2d(
            high_dim_neighbourhood_positions,
            low_dim_neighbourhood_positions,
            bins=n
        )

        # ------------------------------------------------------------------------------------
        # 4. Extract bin indices of records in co-ranking matrix.
        # ------------------------------------------------------------------------------------

        # self._*_dim_ranking specifies the rank/neighbourhood distance in jumps between
        # records x (row) and y (column) - i. e. element (x, y), record y, is record x's
        # self._*_dim_ranking[x, y]-th neighbour.

        # Compile dataframe with information on bin index for record pair.
        # Note that high_dim_neighbour_rank represents y-axis, low_dim_neighbour_offset the x-axis in co-ranking
        # matrix.
        index_vals: list = list(range(n))
        record_indices: np.ndarray = np.asarray([element for element in itertools.product(index_vals, index_vals)])
        record_bin_indices: pd.DataFrame = pd.DataFrame({
            "source": record_indices[:, 0],
            "neighbour": record_indices[:, 1],
            "high_dim_neighbour_rank": high_dim_neighbourhood_positions,
            "low_dim_neighbour_rank": low_dim_neighbourhood_positions
        })

        # Exclude auto-referential records (i. e. rows stating that record x is record's x closest neighbour).
        return Q[1:, 1:], record_bin_indices[record_bin_indices.source != record_bin_indices.neighbour]

    @staticmethod
    def _calculate_geodesic_ranking(data: np.ndarray, distance_metric: str):
        """
        Calculates neighbourhood ranking in data based on geodesic distances.
        :param data:
        :param distance_metric:
        :return:
        """
        k = 2
        is_connex = False
        while not is_connex:
            knn = sklearn.neighbors.NearestNeighbors(n_neighbors=k, metric=distance_metric)
            knn.fit(data)
            M = knn.kneighbors_graph(data, mode='distance')
            graph = nx.from_scipy_sparse_matrix(M)
            is_connex = nx.is_connected(graph)
            k += 1
        distances = nx.all_pairs_dijkstra_path_length(graph, cutoff=None, weight='weight')
        distances = np.array(
            [np.array(a.items())[:, 1] for a in np.array(distances.items())[:, 1]])

        distances = (distances + distances.T) / 2

        # Generate rankings from distances.
        return distances.argsort(axis=1).argsort(axis=1)

    @staticmethod
    def generate_coranking_matrix_deprecated(
            high_dimensional_data: np.ndarray,
            low_dimensional_data: np.ndarray,
            distance_metric: str,
            high_dim_neighbourhood_ranking: np.ndarray = None
    ):
        """
        Calculates coranking matrix (might be used for calculating matrix only once if several coranking-based metrics
        are to be used).
        Uses code from https://github.com/samueljackson92/coranking/. Adaption: Parameter for choice of distance metric,
        miscellaneous refactoring.
        :param high_dimensional_data:
        :param low_dimensional_data:
        :param distance_metric: One of the distance metrics supported by scipy's cdist().
        :param high_dim_neighbourhood_ranking: Ranking of neighbourhood similarities. Calculated if none is supplied. If
        supplied, high_dimensional_data is not used.
        :return: Coranking matrix as 2-dim. ndarry.
        """

        n, m = high_dimensional_data.shape

        # Calculate ranking of neighbourhood similarities for high- and low-dimensional datasets.
        high_ranking = high_dim_neighbourhood_ranking if high_dim_neighbourhood_ranking is not None else \
            CorankingMatrix._generate_neighbourhood_matrix(high_dimensional_data, distance_metric)
        low_ranking = \
            CorankingMatrix._generate_neighbourhood_matrix(low_dimensional_data, distance_metric)

        # Aggregate coranking matrix.
        Q, xedges, yedges = np.histogram2d(high_ranking.flatten(), low_ranking.flatten(), bins=n)

        # Remove rankings which correspond to themselves, return coranking matrix.
        return Q[1:, 1:]

    def matrix(self):
        """
        Returns coranking matrix.
        :return: Coranking matrix.
        """
        return self._matrix

    def calculate_intrusion(self, k_values: list):
        """
        This method allows to compute the fraction of intrusion.

        Implementation based on https://github.com/gdkrmr/coRanking .

        :param Q: The coranking matrix
        :param k_values: The neighbourhood sizes to sample.
        :returns List of intrusion values for each k.
        """

        Q = self._matrix

        # We retrieve the number of points
        n = Q.shape[0]
        # We compute the values
        vals = Q

        measures = []
        for k in k_values:
            # We compute the mask.
            mask = np.zeros([n, n])
            mask[:k, :k] = np.triu(np.ones([k, k]))
            # We compute the normalization constant
            norm = k * (n + 1.)
            # We finally compute the measures
            measures.append((vals * mask).sum() / float(norm))

        return measures

    def calculate_extrusion(self, k_values: list):
        """
        This method allows to compute the fraction of extrusion.

        Implementation based on https://github.com/gdkrmr/coRanking .

        :param Q: The coranking matrix
        :param k_values: The neighbourhood sizes to sample.
        :returns List of extrusion values for each k.
        """

        Q = self._matrix

        # We retrieve the number of points
        n = Q.shape[0]
        # We compute the values
        vals = Q

        # We compute the mask
        mask = np.zeros([n, n])

        measures = []
        for k in k_values:
            mask[:k, :k] = np.tril(np.ones([k, k]))
            # We compute the normalization constant.
            norm = k * (n + 1.)
            # We finally compute the measures.
            measures.append((vals * mask).sum() / float(norm))

            # Reset mask.
            mask[:k, :k] = 0.

        return measures

    def high_dimensional_neighbourhood_ranking(self):
        """
        Returns ranking of neighbours in high-dimensional space.
        :return:
        """

        return self._high_dim_ranking

    def low_dimensional_neighbourhood_ranking(self):
        """
        Returns ranking of neighbours in low-dimensional space.
        :return:
        """

        return self._low_dim_ranking

    def create_pointwise_coranking_matrix_generator(self, indices: list = None):
        """
        Creates a generator yielding one pointwise
        (see section 4 in http://www.cs.rug.nl/biehl/Preprints/2012-esann-quality.pdf) coranking matrix Q per record.
        Applied to all records with index i in indices.
        :param indices:
        :return:
        """

        n, m = self._matrix.shape
        indices = indices if indices is not None else [i for i in range(0, n)]

        # Create pointwise coranking matrix for each index.
        for i in indices:
            # Initialize ranking matrices.
            low_dim_nh_ranking_i = np.zeros(self._low_dim_ranking.shape)
            high_dim_nh_ranking_i = np.zeros(self._high_dim_ranking.shape)

            # Copy row i.
            low_dim_nh_ranking_i[i] = self._low_dim_ranking[i]
            high_dim_nh_ranking_i[i] = self._high_dim_ranking[i]

            # Calculate coranking matrix.
            Q, xedges, yedges = np.histogram2d(
                high_dim_nh_ranking_i.flatten(),
                low_dim_nh_ranking_i.flatten(),
                bins=n
            )

            # Remove rankings which correspond to themselves, return coranking matrix.
            yield Q[1:, 1:]

    @staticmethod
    def compute_pairwise_displacement_data(
            original_distance_matrices_file_path: str,
            original_neighbour_ranking_file_path: str,
            low_dim_projection_data: np.ndarray
    ) -> pd.DataFrame:
        """
        Computes pairwise displacement data, i. e. data needed for Shepard diagram (pairwise distances between records
        in high- and low-dimensional space) and coranking matrix for this low-dimensional embedding.
        :param original_distance_matrices_file_path:
        :param original_neighbour_ranking_file_path:
        :param low_dim_projection_data:
        :return: Dict of pd.DataFrames with records like [
            record index 1,
            record index 2,
            metric,
            high dim distance,
            low dim distance,
            high dim neighbour rank,
            low dim neighbour rank
        ].
        """

        # Get number of records.
        n: int = low_dim_projection_data.shape[0]

        ###############################################
        # 1. Load files.
        ###############################################

        with open(original_neighbour_ranking_file_path, "rb") as file:
            original_neighbour_ranking: dict = pickle.load(file)
        with open(original_distance_matrices_file_path, "rb") as file:
            original_distance_matrices: dict = pickle.load(file)
        distance_metrics = list(original_distance_matrices.keys())

        ###############################################
        # 2. Compute coranking and distance matrices
        # for low-dimensional dataset.
        ###############################################
        # Note that this is not feasible for bigger datasets - but storing distance and coranking matrices for every

        pairwise_displacement_data: pd.DataFrame = pd.DataFrame()
        for metric in distance_metrics:
            # 1. Gather co-ranking matrix information.
            df = CorankingMatrix(
                high_dimensional_data=original_distance_matrices[metric],
                low_dimensional_data=low_dim_projection_data,
                distance_metric=metric,
                high_dimensional_neighbourhood_ranking=original_neighbour_ranking[metric]
            ).record_bin_indices
            df["metric"] = metric

            # 2. Gather distance values, merge with existing dataframes (note that index sequence is identical, since
            # both refer to the distance matrix in its original form + .flatten()).
            hd_distances = original_distance_matrices[metric]
            ld_distances = cdist(low_dim_projection_data, low_dim_projection_data, metric)

            # 3. Discard auto-referential entries (see https://stackoverflow.com/a/46736275) & append distance values to
            # dataframe.
            df["high_dim_distance"] = np.delete(
                hd_distances.flatten(), range(0, hd_distances.size, len(hd_distances) + 1), 0
            )
            df["low_dim_distance"] = np.delete(
                ld_distances.flatten(), range(0, ld_distances.size, len(ld_distances) + 1), 0
            )

            # 4. Fuse dataframes separated by metric into one.
            pairwise_displacement_data = pd.concat([pairwise_displacement_data, df])

        return pairwise_displacement_data
