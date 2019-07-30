import csv
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_swiss_roll
from sklearn.neighbors import kneighbors_graph

from data_generation.datasets import InputDataset


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

    def persist_records(self, directory: str):
        filepath = directory + '/swiss_roll_records.csv'

        if not os.path.isfile(filepath):
            with open(filepath, mode='w') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                # Write header line.
                header_line = ["record_name", "target_label"]
                header_line.extend([i for i in range(0, len(self._data["features"][0]))])
                csv_writer.writerow(header_line)

                # Append records to .csv.
                for i, features in enumerate(self._data["features"]):
                    # Use index as record name, since records are anonymous.
                    line = [i, self._data["labels"][i]]
                    line.extend(features)
                    csv_writer.writerow(line)
