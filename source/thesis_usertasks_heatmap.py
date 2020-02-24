import pandas as pd
import itertools
import seaborn as sns
import matplotlib.pyplot as plt


components: list = ["C" + str(i).zfill(2) for i in range(1, 14)]
tasks: list = [
    *["T1." + str(i) for i in range(1, 9)],
    *["T2." + str(i) for i in range(1, 5)],
    *["T3." + str(i) for i in range(1, 4)]
]
matrix: pd.DataFrame = pd.DataFrame(list(itertools.product(components, tasks)), columns=["Component", "Task"])
matrix["interaction"] = 0
matrix.loc[
    (matrix.Component == "C01") &
    (matrix.Task.isin(("T1.2", "T2.1", "T2.3", "T3.3"))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == "C02") &
    (matrix.Task.isin(("T1.2", "T2.1", "T2.3", "T3.1", "T3.3"))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == "C03") &
    (matrix.Task.isin(("T1.2", "T2.1", "T2.3", "T3.1", "T3.3"))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == "C04") &
    (matrix.Task.isin(("T1.1", "T1.2", "T3.3"))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == "C05") &
    (matrix.Task.isin(("T2.2", "T2.3", "T2.4"))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == "C06") &
    (matrix.Task.isin(("T3.2",))),
    "interaction"
] = 1

matrix.loc[
    (matrix.Component == "C07") &
    (matrix.Task.isin(("T1.2",))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == "C08") &
    (matrix.Task.isin(("T1.7",))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == "C09") &
    (matrix.Task.isin(("T1.5", "T1.6"))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == "C10") &
    (matrix.Task.isin(("T1.4", "T1.5", "T1.6"))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == "C11") &
    (matrix.Task.isin(("T1.3",))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == "C12") &
    (matrix.Task.isin(("T1.3",))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == "C13") &
    (matrix.Task.isin(("T1.8",))),
    "interaction"
] = 1

# Reshape for seaborn.heatmap().
matrix = matrix.pivot(columns="Component", index="Task", values="interaction")
matrix = matrix.rename(columns={col: col.replace("C0", "C") for col in matrix.columns})
ax = sns.heatmap(
    matrix, annot=False, fmt="g", linewidths=1, cmap=sns.cubehelix_palette(8, start=0, dark=0, light=.95), cbar=False
)
ax.set_xlabel("Task", fontsize=20, labelpad=20)
ax.set_ylabel("Component", fontsize=20, labelpad=20)
ax.tick_params(labelsize=15)
plt.show()
