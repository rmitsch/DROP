# todo
#   - recompute separability
#   - plot metrics vs. ratings

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

dirpath: str = "/home/raphael/Development/data/TALE/TALE-study"
df: pd.DataFrame = pd.concat([
    pd.read_pickle(os.path.join(dirpath, f)).reset_index().drop(
        columns=["metric", "num_records"], errors="ignore"
    )
    for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))
])
df = df[df.rating != 0]

with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
    print(df.head())
    print(len(df))

for metric in [
    ("r_nx", "Neighbourhood stability (r_nx)"),
    ("stress", "Distance (Kruskal's Stress)"),
    ("target_domain_performance", "Relative predictive power"),
    ("separability_metric", "Cluster purity (Silhouette)")
]:
    ax = sns.regplot(x=metric[0], y="rating", data=df, ci=90, fit_reg=True)
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


