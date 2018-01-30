import threading
import numpy
from sklearn.manifold import TSNE
import time
from MulticoreTSNE import MulticoreTSNE


class TSNEThread(threading.Thread):
    """
    Class calculating a t-SNE model for the given distance matrix with the specified parametrizations.
    """

    def __init__(
            self,
            results: list,
            distance_matrices: dict,
            parameter_sets: list,
            high_dimensional_data: numpy.ndarray):
        """
        Initializes thread instance that will calculate the low-dimensional representation of the specified distance
        matrices applying t-SNE.
        :param results:
        :param distance_matrices:
        :param parameter_sets:
        :param high_dimensional_data:
        """
        threading.Thread.__init__(self)

        self.distance_matrices = distance_matrices
        self.parameter_sets = parameter_sets
        self.results = results
        self.high_dimensional_data = high_dimensional_data

    def run(self):
        """
        Runs thread and calculates all t-SNE models specified in parameter_sets.
        :return: List of 2D-ndarrays containing coordinates of instances in low-dimensional space.
        """

        for parameter_set in self.parameter_sets:
            # Calculate t-SNE.
            start = time.time()
            low_dimensional_projection = MulticoreTSNE(
                n_components=parameter_set["n_components"],
                perplexity=parameter_set["perplexity"],
                early_exaggeration=parameter_set["early_exaggeration"],
                learning_rate=parameter_set["learning_rate"],
                n_iter=parameter_set["n_iter"],
                # min_grad_norm=parameter_set["min_grad_norm"],
                angle=parameter_set["angle"],
                # Always set metric to 'precomputed', since distance matrices are calculated previously. If other
                # metrics are desired, the corresponding preprocessing step has to be extended.
                metric='precomputed',
                method='barnes_hut' if parameter_set["n_components"] < 4 else 'exact',
                # Set n_jobs to 1, since we parallelize at a higher level by splitting up model parametrizations amongst
                # threads.
                n_jobs=1
            ).fit_transform(self.distance_matrices[parameter_set["metric"]])

            # Append runtime to set of objectives.
            objectives = {
                "runtime": time.time() - start,
                # todo Calculate trustworthiness here.
                "trustworthiness": 1
            }

            # Store parameter set, objective set and low dimensional projection in globally shared object.
            self.results.append({
                "parameter_set": parameter_set,
                "objectives": objectives,
                "low_dimensional_projection": low_dimensional_projection
            })
