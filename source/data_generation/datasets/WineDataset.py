import csv
import os
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from data_generation.datasets import InputDataset


class WineDataset(InputDataset):
    """
    Loads wine dataset from sklearn.
    """

    def __init__(self, data=None, preprocessed_features=None, classification_accuracy=None):
        super().__init__(
            data=data,
            preprocessed_features=preprocessed_features,
            classification_accuracy=classification_accuracy
        )

    def _load_data(self):
        return load_wine()

    def features(self):
        return self._preprocessed_features

    def labels(self):
        return self._data.target

    def _preprocess_features(self):
        # Scale features.
        return StandardScaler().fit_transform(self._data.data)

    def persist_records(self, directory: str):
        filepath = directory + '/wine_records.csv'

        if not os.path.isfile(filepath):
            with open(filepath, mode='w') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                # Write header line.
                header_line = ["record_name", "target_label"]
                header_line.extend(self._data["feature_names"])
                csv_writer.writerow(header_line)

                # Append records to .csv.
                for i, features in enumerate(self._data.data):
                    # Use index as record name, since records are anonymous.
                    line = [i, self._data.target[i]]
                    line.extend(features)
                    csv_writer.writerow(line)
