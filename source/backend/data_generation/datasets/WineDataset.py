import csv
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from backend.data_generation.datasets import InputDataset


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
        with open(directory + '/wine_records.csv', mode='a') as csv_file:
            # Append as not to overwrite needed data. .csv won't be usable w/o rectification after appending, but we
            # assume some manual postprocessing to be preferrable to data loss due to carelessness.
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            # Write header line.
            header_line = ["record_name", "target_label"]
            header_line.extend(self._data["feature_names"])
            csv_writer.writerow(header_line)

            # Append records to .csv.
            for i, features in enumerate(self.features()):
                # Use index as record name, since records are anonymous.
                line = [i, self._data.target[i]]
                line.extend(features)
                csv_writer.writerow(line)



