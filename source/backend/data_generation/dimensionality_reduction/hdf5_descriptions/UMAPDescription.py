from tables import *
from .DRDescription import DRDescription


class UMAPDescription(DRDescription):
    """
    Class used as representative for storing UMAP parameter sets in .h5 files.
    """

    # Hyperparameter.
    n_components = Int32Col(pos=9)
    n_neighbors = Int32Col(pos=10)
    n_epochs = Int32Col(pos=11)
    learning_rate = Float64Col(pos=12)
    min_dist = Float64Col(pos=13)
    local_connectivity = Int32Col(pos=14)
    metric = StringCol(20, pos=15)
