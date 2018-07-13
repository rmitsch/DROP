import numpy
from MulticoreTSNE import MulticoreTSNE

from backend.data_generation import InputDataset, sklearn
from backend.objectives.topology_preservation_objectives import *
from backend.objectives.distance_preservation_objectives import *


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
                {"name": "min_grad_norm", type: "numeric"},
                {"name": "angle", type: "numeric"},
                {"name": "metric", type: "categorical"}
            ],
            "hdf5_description": None
        },
        "SVD": {
            "parameters": [
                {"name": "n_components", type: "numeric"}
            ],
            "hdf5_description": None
        },
        "UMAP": {
            "parameters": [
                {"name": "n_components", type: "numeric"}
            ],
            "hdf5_description": None
        }
    }

    def __init__(self, high_dim_data: numpy.ndarray, dim_red_kernel: str):
        """
        Initializes new DimensionalityReductionKernel.
        :param high_dim_data:
        :param dim_red_kernel:
        """
        self._high_dim_data = high_dim_data
        self._dim_red_kernel = dim_red_kernel

    def run(self, parameter_set: dict):
        """
        Applies chosen DR kernel on specified high dimensional data set with this parametrization.
        :param parameter_set:
        :return: Low-dimensional projection of high-dimensional data.
        """
        pass
