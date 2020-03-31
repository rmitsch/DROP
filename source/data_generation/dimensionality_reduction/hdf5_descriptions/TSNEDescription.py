from tables import *
from .DRDescription import DRDescription


class TSNEDescription(DRDescription):
    """
    Class used as representative for storing TSNE parameter sets in .h5 files.
    """

    # Hyperparameter.
    n_components = Int32Col(pos=9)
    perplexity = Int32Col(pos=10)
    early_exaggeration = Float64Col(pos=11)
    learning_rate = Float64Col(pos=12)
    n_iter = UInt16Col(pos=13)
    angle = Float64Col(pos=14)
