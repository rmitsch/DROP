import pandas as pd
import itertools
import seaborn as sns
import matplotlib.pyplot as plt


components: list = ["HE", "SSP", "HH", "RP", "GT", "HHIG", "HMI", "HHIL", "SP", "RT", "SD", "CRM", "RC"]

tasks: list = [
    *["$T_{INSP}$." + str(i) for i in range(1, 9)],
    *["$T_{NAV-PS}$." + str(i) for i in range(1, 5)],
    *["$T_{INT-PS}$." + str(i) for i in range(1, 4)]
]
matrix: pd.DataFrame = pd.DataFrame(list(itertools.product(components, tasks)), columns=["Component", "Task"])
matrix["interaction"] = 0
matrix.loc[
    (matrix.Component == components[0]) &
    (matrix.Task.isin(("$T_{INSP}$.2", "$T_{NAV-PS}$.1", "$T_{NAV-PS}$.3", "$T_{INT-PS}$.3"))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == components[1]) &
    (matrix.Task.isin(("$T_{INSP}$.2", "$T_{NAV-PS}$.1", "$T_{NAV-PS}$.3", "$T_{INT-PS}$.1", "$T_{INT-PS}$.3"))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == components[2]) &
    (matrix.Task.isin(("$T_{INSP}$.2", "$T_{NAV-PS}$.1", "$T_{NAV-PS}$.3", "$T_{INT-PS}$.1", "$T_{INT-PS}$.3"))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == components[3]) &
    (matrix.Task.isin(("$T_{INSP}$.1", "$T_{INSP}$.2", "$T_{INT-PS}$.3"))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == components[4]) &
    (matrix.Task.isin(("$T_{NAV-PS}$.2", "$T_{NAV-PS}$.3", "$T_{NAV-PS}$.4"))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == components[5]) &
    (matrix.Task.isin(("$T_{INT-PS}$.2",))),
    "interaction"
] = 1

matrix.loc[
    (matrix.Component == components[6]) &
    (matrix.Task.isin(("$T_{INSP}$.2",))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == components[7]) &
    (matrix.Task.isin(("$T_{INSP}$.7",))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == components[8]) &
    (matrix.Task.isin(("$T_{INSP}$.5", "$T_{INSP}$.6"))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == components[9]) &
    (matrix.Task.isin(("$T_{INSP}$.4", "$T_{INSP}$.5", "$T_{INSP}$.6"))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == components[10]) &
    (matrix.Task.isin(("$T_{INSP}$.3",))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == components[11]) &
    (matrix.Task.isin(("$T_{INSP}$.3",))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == components[12]) &
    (matrix.Task.isin(("$T_{INSP}$.8",))),
    "interaction"
] = 1


def assign_color(row: pd.Series) -> str:
    """
    Assigns color to interaction w.r.t. task group and interaction yes/no.
    :param row: Interaction.
    :return: Color for cell in interaction matrix plot.
    """

    if row.interaction:
        if "INSP" in row.Task:
            return "lightblue"

        elif "INT-PS" in row.Task:
            return "green"

        else:
            return "orange"

    return "white"


matrix["color"] = matrix.apply(lambda x: assign_color(x), axis=1)

matrix.loc[
    (matrix.Task.isin(("$T_{INSP}$." + str(i) for i in range(1, 9))) & matrix.interaction),
    "interaction"
] *= 0.25

matrix.loc[
    (matrix.Task.isin(("$T_{INT-PS}$." + str(i) for i in range(1, 5))) & matrix.interaction),
    "interaction"
] *= 0.5

matrix.loc[
    (matrix.Task.isin(("$T_{NAV-PS}$." + str(i) for i in range(1, 5))) & matrix.interaction),
    "interaction"
] *= 0.75

# Reshape for seaborn.heatmap().
matrix = matrix.pivot(columns="Component", index="Task", values="interaction")
# ax = sns.heatmap(
#     matrix,
#     annot=False,
#     fmt="g",
#     linewidths=0.05,
#     cmap=sns.color_palette("Paired", 4),
#     cbar=False,
#     mask=matrix == 0,
#     linecolor="lightgrey"
# )
# ax.set_xlabel("Component", fontsize=30, labelpad=20)
# ax.set_ylabel("Task", fontsize=30, labelpad=20)
# ax.tick_params(labelsize=25)
# plt.show()

print(sns.__version__)

exp_col = "Experience with dim. red."
exp_df = pd.DataFrame([0, 0, 2, 2, 2, 1, 1, 0, 1, 2, 2.5, 2.5, 1, 2.5, 2.5], columns=[exp_col])
ax = sns.histplot(data=exp_df, x=exp_col, bins=[0, 1, 2, 3])
ax.set_xlabel("Experience with dimensionality reduction", fontsize=20, labelpad=20)
ax.set_ylabel("Count of participants", fontsize=20, labelpad=20)
ax.tick_params(labelsize=15)
plt.show()
