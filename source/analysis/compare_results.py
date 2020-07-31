import os
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import pickle
import pandas as pd


if __name__ == '__main__':
    base_path: str = "/home/raphael/Development/data/TALE/evaluation/"
    results: List[Dict] = []

    for f in os.listdir(base_path):
        if os.path.isfile(os.path.join(base_path, f)) and f.endswith(".pkl"):
            with open(base_path + f, 'rb') as handle:
                estimator, metrics, selected_feature_data, test_feats, test_labels = pickle.load(handle)
            metrics = metrics[metrics.model == "lgbm"].mean()
            metrics["set"] = f[:-4]
            results.append(metrics.to_dict())

    results: pd.DataFrame = pd.DataFrame(results)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(results)
    results.sort_values("mean_absolute_error").plot.bar(
        x="set", y="mean_absolute_error", grid=True
    )
    plt.tight_layout()
    plt.show()
