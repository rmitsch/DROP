import csv
import os
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from backend.data_generation.datasets import InputDataset


class MNISTDataset(InputDataset):
    """
    Load MNIST datasets using sklearn.
    """

    def __init__(self):
        super().__init__()

    def _load_data(self):
        # Fetch MNIST dataset.
        mnist_data = fetch_mldata('MNIST original', data_home="../data")
        reduced_mnist_data = {
            "features": None,
            "labels": None
        }

        # MNIST with 70k entries requires too much memory for distance matrices for this machine.
        # Hence: We pick a subset of 10k as stratified sample.
        # todo Make selection deterministic.
        for train_indices, test_indices in StratifiedShuffleSplit(
                n_splits=1,
                test_size=1 / 200
        ).split(mnist_data.data, mnist_data.target):
            reduced_mnist_data["features"] = mnist_data.data[test_indices].astype(int)
            reduced_mnist_data["labels"] = mnist_data.target[test_indices].astype(int)

        return reduced_mnist_data

    def features(self):
        return self._data["features"]

    def labels(self):
        return self._data["labels"]

    def _preprocess_features(self):
        # Scale features.
        return StandardScaler().fit_transform(self._data["features"])

    def persist_records(self, directory: str):
        filepath = directory + '/mnist_records.csv'

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
