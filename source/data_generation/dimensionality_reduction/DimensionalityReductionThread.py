import threading
import time
import numpy as np
from data_generation import InputDataset
from objectives.topology_preservation_objectives import *
from objectives.distance_preservation_objectives import *
from .DimensionalityReductionKernel import DimensionalityReductionKernel


class DimensionalityReductionThread(threading.Thread):
    """
    Thread executing DR method of choice on a specific dataset with a set of parametrizations.
    """

    def __init__(
            self,
            results: list,
            distance_matrix: np.ndarray,
            parameter_sets: list,
            input_dataset: InputDataset,
            high_dimensional_neighbourhood_ranking: np.ndarray,
            dim_red_kernel_name: str
    ):
        """
        Initializes thread instance that will calculate the low-dimensional representation of the specified distance
        matrices applying the chosen DR method.
        :param results:
        :param distance_matrix:
        :param parameter_sets:
        :param input_dataset:
        :param high_dimensional_neighbourhood_ranking: Neighbourhood rankings in original high-dimensional space. Dict.
        with one entry per distance metric.
        :param dim_red_kernel_name: Name of dimensionality reduction algorithm to apply.
        """
        threading.Thread.__init__(self)

        self._distance_matrix: np.ndarray = distance_matrix
        self._parameter_sets: list = parameter_sets
        self._results: list = results
        self._input_dataset: InputDataset = input_dataset
        self._high_dimensional_neighbourhood_ranking: np.ndarray = high_dimensional_neighbourhood_ranking
        self._dim_red_kernel: DimensionalityReductionKernel = DimensionalityReductionKernel(dim_red_kernel_name)

    def run(self):
        """
        Runs thread and calculates all t-SNE models specified in parameter_sets.
        :return: List of 2D ndarrays containing coordinates of instances in low-dimensional space.
        """

        ###################################################
        # 1. Calculate embedding for each distance metric.
        ###################################################

        for parameter_set in self._parameter_sets:
            # Calculate t-SNE. Supress output while doing so.
            start: float = time.time()
            low_dimensional_projection: np.ndarry = self._dim_red_kernel.run(
                high_dim_data=self._distance_matrix,
                parameter_set=parameter_set
            )

            ###################################################
            # 2. Calculate objectives.
            ###################################################

            # Start measuring runtime.
            runtime: float = time.time() - start

            ############################################
            # 2. a. Topology-based metrics.
            ############################################

            # Create coranking matrix for topology-based objectives.
            coranking_matrix: CorankingMatrix = CorankingMatrix(
                low_dimensional_data=low_dimensional_projection,
                high_dimensional_neighbourhood_ranking=self._high_dimensional_neighbourhood_ranking
            )

            r_nx: float = CorankingMatrixQualityCriterion(
                low_dimensional_data=low_dimensional_projection,
                coranking_matrix=coranking_matrix
            ).compute()

            b_nx: float = CorankingMatrixBehaviourCriterion(
                low_dimensional_data=low_dimensional_projection,
                coranking_matrix=coranking_matrix
            ).compute()

            # Pointwise Q_nx -> q_nx.
            q_nx_i: float = PointwiseCorankingMatrixQualityCriterion(
                low_dimensional_data=low_dimensional_projection,
                coranking_matrix=coranking_matrix
            ).compute()

            ############################################
            # 2. b. Distance-based metrics.
            ############################################

            stress: float = Stress(
                high_dimensional_data=self._distance_matrix,
                low_dimensional_data=low_dimensional_projection,
                use_geodesic_distances=False
            ).compute()

            ############################################
            # 2. c. Information-preserving metrics.
            ############################################

            rtdp: float = self._input_dataset.compute_relative_target_domain_performance(
                features=low_dimensional_projection
            )

            ############################################
            # 2. d. Separability metrics.
            ############################################

            separability_metric: float = self._input_dataset.compute_separability_metric(
                features=low_dimensional_projection
            )

            ###################################################
            # 3. Collect data, terminate.
            ###################################################

            # Append runtime to set of objectives.
            objectives: dict = {
                "runtime": runtime,
                "r_nx": r_nx,
                "b_nx": b_nx,
                "stress": stress,
                "target_domain_performance": rtdp,
                "separability_metric": separability_metric,
                "pointwise_quality_values": q_nx_i
            }

            # Store parameter set, objective set, low dimensional projection and pointwise quality criterion values in
            # globally shared object.
            self._results.append({
                "parameter_set": parameter_set,
                "objectives": objectives,
                "low_dimensional_projection": low_dimensional_projection
            })
