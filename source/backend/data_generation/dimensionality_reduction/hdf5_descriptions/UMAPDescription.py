from tables import *
from .DRDescription import DRDescription


class UMAPDescription(DRDescription):
    """
    Class used as representative for storing UMAP parameter sets in .h5 files.
    """

    # Hyperparameter.
    n_components = Int32Col(pos=9)
