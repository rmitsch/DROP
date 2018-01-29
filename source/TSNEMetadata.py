from tables import *


class TSNEMetadata(IsDescription):
    """
    Class used as representative for storing generated parameter sets in .h5 files.
    """

    id = Int16Col(pos=1)

    # Hyperparameter.
    n_components = Int16Col(pos=2)
    perplexity = Int16Col(pos=3)
    early_exaggeration = Float32Col(pos=4)
    learning_rate = Float16Col(pos=5)
    n_iter = UInt8Col(pos=6)
    # min_grad_norm = Float16Col(pos=7)
    angle = Float16Col(pos=8)
    metric = StringCol(20, pos=9)

    # Objectives for faithfulness of dimensionality-reduced projection.
    trustworthiness = Float16Col(pos=10)

    # Other quality measures.
    runtime = Int8Col(pos=11)

