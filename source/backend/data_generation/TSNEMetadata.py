from tables import *


class TSNEMetadata(IsDescription):
    """
    Class used as representative for storing generated parameter sets in .h5 files.
    """

    id = Int16Col(pos=1)
    num_records = Int16Col(pos=2)

    # Hyperparameter.
    n_components = Int32Col(pos=3)
    perplexity = Int32Col(pos=4)
    early_exaggeration = Float64Col(pos=5)
    learning_rate = Float64Col(pos=6)
    n_iter = UInt16Col(pos=7)
    # min_grad_norm = Float16Col(pos=7)
    angle = Float64Col(pos=8)
    metric = StringCol(20, pos=9)

    # Objectives for faithfulness of dimensionality-reduced projection.
    r_nx = Float32Col(pos=10)
    b_nx = Float32Col(pos=11)
    stress = Float32Col(pos=12)
    classification_accuracy = Float32Col(pos=13)
    separability_metric = Float32Col(pos=14)
    runtime = Int8Col(pos=15)

