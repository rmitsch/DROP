# todo
#   - recompute separability
#   - plot metrics vs. ratings
from typing import List

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt


def read_user_scores(dirpath: str) -> pd.DataFrame:
    """
    Reads user score pickles from disk and combines them into one dataframe.
    :param dirpath: Path to directory holding pickles with user data.
    :return: Combined dataframe with user scores.
    """

    user_scores: List[pd.DataFrame] = []

    for f in os.listdir(dirpath):
        if os.path.isfile(os.path.join(dirpath, f)):
            us_df: pd.DataFrame = pd.read_pickle(os.path.join(dirpath, f)).reset_index().drop(
                columns=["metric", "num_records"], errors="ignore"
            )
            us_df["dataset"] = f.split("_")[2]
            user_scores.append(us_df)

    user_scores: pd.DataFrame = pd.concat(user_scores)

    return user_scores[user_scores.rating != 0]


if __name__ == '__main__':
    df: pd.DataFrame = read_user_scores("/home/raphael/Development/data/TALE/TALE-study")

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(df.head())
        print(len(df))

    for metric in [
        ("r_nx", "Neighbourhood stability ($NH_{AUC}$)"),
        ("stress", "Distance (Kruskal's Stress)"),
        ("target_domain_performance", "Relative predictive power"),
        ("separability_metric", "Cluster purity (Silhouette)")
    ]:
        ax = sns.regplot(x=metric[0], y="rating", data=df, ci=95, fit_reg=True)
        ax.set_title(metric[1] + " vs. perceptual rating \n")
        plt.xlabel(metric[1])
        plt.ylabel("Perceptual rating")

        ax.text(
            x=0.5,
            y=1.02,
            s=(
                'corr_pearson = ' + str(df[[metric[0], "rating"]].corr(method="pearson").values[0, 1])[:5] + ", " +
                'corr_kendall = ' + str(df[[metric[0], "rating"]].corr(method="kendall").values[0, 1])[:5] + ", " +
                'corr_spearman = ' + str(df[[metric[0], "rating"]].corr(method="spearman").values[0, 1])[:5]
            ),
            fontsize=8,
            alpha=0.75,
            ha='center',
            va='bottom',
            transform=ax.transAxes
        )
        plt.show()


