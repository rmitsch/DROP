from tables import *
from .DRDescription import DRDescription


class SVDDescription(DRDescription):
    """
    Class used as representative for storing SVD parameter sets in .h5 files.
    """

    # Hyperparameter.
    n_components = Int32Col(pos=9)
