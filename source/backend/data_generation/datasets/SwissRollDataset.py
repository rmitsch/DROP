from backend.data_generation.datasets import InputDataset
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_swiss_roll
from sklearn.neighbors import kneighbors_graph


class SwissRollDataset(InputDataset):
    """
    Generate swiss roll dataset using sklearn.
    Following tutorial here:
    http://scikit-learn.org/stable/auto_examples/cluster/plot_ward_structured_vs_unstructured.html#sphx-glr-auto-examples-cluster-plot-ward-structured-vs-unstructured-py
    """

    def __init__(self):
        super().__init__()

    def _load_data(self):
        data = {
            "features": None,
            "labels": None
        }

        ####################################
        # 1. Generate swiss roll dataset.
        ####################################

        n_samples = 1000
        noise = 0.05
        X, _ = make_swiss_roll(n_samples, noise)
        # Make it thinner
        X[:, 1] *= .5
        data["features"] = X

        ####################################
        # 2. Assign class labels through
        # clustering.
        ####################################

        # Define the structure A of the data. Here with the 10 nearest neighbours.
        connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)

        # Cluster using connectivity graph.
        data["labels"] = AgglomerativeClustering(
            n_clusters=6,
            connectivity=connectivity,
            linkage='ward'
        ).fit(X).labels_

        return data

    def features(self):
        return self._data["features"]

    def labels(self):
        return self._data["labels"]

    def _preprocess_features(self):
        # No preprocessing done here.
        return self._data["features"]
