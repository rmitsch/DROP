from tables import *
from .DRDescription import DRDescription


class SVDDescription(DRDescription):
    """
    Class used as representative for storing SVD parameter sets in .h5 files.
    """

    # Hyperparameter.
    n_components = Int32Col(pos=9)
    n_iter = Int32Col(pos=10)
    metric = StringCol(20, pos=11)
