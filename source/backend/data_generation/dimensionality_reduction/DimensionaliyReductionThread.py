import threading
import time

from MulticoreTSNE import MulticoreTSNE
from sklearn.preprocessing import StandardScaler

from backend.data_generation import InputDataset, sklearn
from backend.objectives.topology_preservation_objectives import *
from backend.objectives.distance_preservation_objectives import *
from .DimensionalityReductionKernel import DimensionalityReductionKernel


class DimensionalityReductionThread(threading.Thread):
    """
    Thread executing DR method of choice on a specific dataset with a set of parametrizations.
    """

    def __init__(
            self,
            results: list,
            distance_matrices: dict,
            parameter_sets: list,
            input_dataset: InputDataset,
            high_dimensional_neighbourhood_rankings: dict,
            dim_red_kernel_name: str
    ):
        """
        Initializes thread instance that will calculate the low-dimensional representation of the specified distance
        matrices applying the chosen DR method.
        :param results:
        :param distance_matrices:
        :param parameter_sets:
        :param input_dataset:
        :param high_dimensional_neighbourhood_rankings: Neighbourhood rankings in original high-dimensional space. Dict.
        with one entry per distance metric.
        :param dim_red_kernel_name: Name of dimensionality reduction algorithm to apply.
        """
        threading.Thread.__init__(self)

        self._distance_matrices = distance_matrices
        self._parameter_sets = parameter_sets
        self._results = results
        self._input_dataset = input_dataset
        self._high_dimensional_neighbourhood_rankings = high_dimensional_neighbourhood_rankings
        self._dim_red_kernel = DimensionalityReductionKernel(dim_red_kernel_name)

    def run(self):
        """
        Runs thread and calculates all t-SNE models specified in parameter_sets.
        :return: List of 2D ndarrays containing coordinates of instances in low-dimensional space.
        """

        ###################################################
        # 1. Calculate embedding for each distance metric.
        ###################################################

        for parameter_set in self._parameter_sets:
            metric = parameter_set["metric"]

            # Calculate t-SNE. Surpress output while doing so.
            start = time.time()
            low_dimensional_projection = self._dim_red_kernel.run(
                high_dim_data=self._distance_matrices[metric],
                parameter_set=parameter_set
            )

            # Scale projection data for later use.
            scaled_low_dim_projection = low_dimensional_projection #StandardScaler().fit_transform(low_dimensional_projection)

            ###################################################
            # 2. Calculate objectives.
            ###################################################

            # Start measuring runtime.
            runtime = time.time() - start

            ############################################
            # 2. a. Topology-based metrics.
            ############################################

            # Create coranking matrix for topology-based objectives.
            coranking_matrix = CorankingMatrix(
                high_dimensional_data=self._distance_matrices[metric],
                low_dimensional_data=low_dimensional_projection,
                distance_metric=metric,
                high_dimensional_neighbourhood_ranking=self._high_dimensional_neighbourhood_rankings[metric]
            )

            # R_nx.
            r_nx = CorankingMatrixQualityCriterion(
                high_dimensional_data=self._distance_matrices[metric],
                low_dimensional_data=low_dimensional_projection,
                distance_metric=metric,
                coranking_matrix=coranking_matrix
            ).compute()

            # B_nx.
            b_nx = CorankingMatrixBehaviourCriterion(
                high_dimensional_data=self._distance_matrices[metric],
                low_dimensional_data=low_dimensional_projection,
                distance_metric=metric,
                coranking_matrix=coranking_matrix
            ).compute()

            # Pointwise Q_nx -> q_nx.
            q_nx_i = PointwiseCorankingMatrixQualityCriterion(
                high_dimensional_data=self._distance_matrices[metric],
                low_dimensional_data=low_dimensional_projection,
                distance_metric=metric,
                coranking_matrix=coranking_matrix
            ).compute()

            ############################################
            # 2. b. Distance-based metrics.
            ############################################

            stress = Stress(
                high_dimensional_data=self._distance_matrices[metric],
                low_dimensional_data=low_dimensional_projection,
                distance_metric=metric,
                use_geodesic_distances=False
            ).compute()

            ############################################
            # 2. c. Information-preserving metrics.
            ############################################

            classification_accuracy = self._input_dataset.calculate_classification_accuracy(
                features=scaled_low_dim_projection
            )

            ############################################
            # 2. d. Separability metrics.
            ############################################

            separability_metric = self._input_dataset.compute_separability_metric(
                features=scaled_low_dim_projection
            )

            ###################################################
            # 3. Collect data, terminate.
            ###################################################

            # Append runtime to set of objectives.
            objectives = {
                "runtime": runtime,
                "r_nx": r_nx,
                "b_nx": b_nx,
                "stress": stress,
                "classification_accuracy": classification_accuracy,
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
