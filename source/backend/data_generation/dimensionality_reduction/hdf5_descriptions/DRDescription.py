from tables import *


class DRDescription(IsDescription):
    """
    Base class used as representatiion for storing generated dimensionality reduction parameter sets in .h5 files.
    Objectives are defined here, all hyperparameters in subclasses.
    """

    id = Int16Col(pos=1)
    num_records = Int16Col(pos=2)

    # Objectives for faithfulness of dimensionality-reduced projection.
    # Same for every DR method.
    r_nx = Float32Col(pos=3)
    b_nx = Float32Col(pos=4)
    stress = Float32Col(pos=5)
    classification_accuracy = Float32Col(pos=6)
    separability_metric = Float32Col(pos=7)
    runtime = Float32Col(pos=8)
