import numpy
from MulticoreTSNE import MulticoreTSNE
from sklearn.decomposition import TruncatedSVD
from . import hdf5_descriptions
import umap


class DimensionalityReductionKernel:
    """
    Represents an instance of a dimensionality reduction method.
    Currently supported: TSNE, SVD and UMAP.
    """
    # Supported dimensionality reduction algorithms and their parameters.
    # Note that the
    DIM_RED_KERNELS = {
        "TSNE": {
            "parameters": [
                {"name": "n_components", type: "numeric"},
                {"name": "perplexity", type: "numeric"},
                {"name": "early_exaggeration", type: "numeric"},
                {"name": "learning_rate", type: "numeric"},
                {"name": "n_iter", type: "numeric"},
                {"name": "angle", type: "numeric"},
                {"name": "metric", type: "categorical"}
            ],
            "hdf5_description": hdf5_descriptions.TSNEDescription
        },
        "SVD": {
            "parameters": [
                {"name": "n_components", type: "numeric"},
                {"name": "n_iter", type: "numeric"}
            ],
            "hdf5_description": hdf5_descriptions.SVDDescription
        },
        "UMAP": {
            "parameters": [
                {"name": "n_components", type: "numeric"},
                {"name": "n_neighbors", type: "numeric"},
                {"name": "n_epochs", type: "numeric"},
                {"name": "learning_rate", type: "numeric"},
                {"name": "min_dist", type: "numeric"},
                {"name": "spread", type: "numeric"},
                {"name": "metric", type: "categorical"}
            ],
            "hdf5_description": hdf5_descriptions.UMAPDescription
        }
    }

    def __init__(self, dim_red_kernel_name: str):
        """
        Initializes new DimensionalityReductionKernel.
        :param dim_red_kernel_name:
        """

        self._dim_red_kernel_name = dim_red_kernel_name
        self._high_dim_data = None
        self._parameter_set = None

    def run(self, high_dim_data: numpy.ndarray, parameter_set: dict):
        """
        Applies chosen DR kernel on specified high dimensional data set with this parametrization.
        :param high_dim_data:
        :param parameter_set:
        :return: Low-dimensional projection of high-dimensional data.
        """

        self._high_dim_data = high_dim_data
        self._parameter_set = parameter_set

        # Calculate low-dim. projection.
        if self._dim_red_kernel_name == "TSNE":
            return MulticoreTSNE(
                n_components=parameter_set["n_components"],
                perplexity=parameter_set["perplexity"],
                early_exaggeration=parameter_set["early_exaggeration"],
                learning_rate=parameter_set["learning_rate"],
                n_iter=parameter_set["n_iter"],
                angle=parameter_set["angle"],
                # Always set metric to 'precomputed', since distance matrices are calculated previously. If other
                # metrics are desired, the corresponding preprocessing step has to be extended.
                metric='precomputed',
                method='barnes_hut' if parameter_set["n_components"] < 4 else 'exact',
                # Set n_jobs to 1, since we parallelize at a higher level by splitting up model parametrizations amongst
                # threads.
                n_jobs=1
            ).fit_transform(high_dim_data)

        elif self._dim_red_kernel_name == "SVD":
            return TruncatedSVD(
                n_components=parameter_set["n_components"],
                n_iter=parameter_set["n_iter"]
            ).fit_transform(high_dim_data)

        elif self._dim_red_kernel_name == "UMAP":
            return umap.UMAP(
                n_components=parameter_set["n_components"],
                n_neighbors=parameter_set["n_neighbors"],
                n_epochs=parameter_set["n_epochs"],
                learning_rate=parameter_set["learning_rate"],
                min_dist=parameter_set["min_dist"],
                spread=parameter_set["spread"],
                # Always set metric to 'precomputed', since distance matrices are calculated previously. If other
                # metrics are desired, the corresponding preprocessing step has to be extended.
                metric='precomputed',
            ).fit_transform(high_dim_data)

        return None
