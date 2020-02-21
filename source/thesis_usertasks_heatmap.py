import pandas as pd
import itertools
import seaborn as sns


components: list = ["C" + str(i) for i in range(1, 12)]
tasks: list = [
    *["T1." + str(i) for i in range(1, 10)],
    *["T2." + str(i) for i in range(1, 5)],
    *["T3." + str(i) for i in range(1, 4)]
]
matrix: pd.DataFrame = pd.DataFrame(list(itertools.product(components, tasks)), columns=["Component", "Task"])
matrix["interaction"] = 0
matrix.loc[
    (matrix.Component == "C1") &
    (matrix.Task.isin(("T1.3", "T2.1", "T2.3", "T3.3"))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == "C2") &
    (matrix.Task.isin(("T1.3", "T2.1", "T2.3", "T3.1", "T3.3"))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == "C3") &
    (matrix.Task.isin(("T1.3", "T2.1", "T2.3", "T3.1", "T3.3"))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == "C4") &
    (matrix.Task.isin(("T1.1", "T1.3", "T3.3"))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == "C5") &
    (matrix.Task.isin(("T2.2", "T2.3"))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == "C6") &
    (matrix.Task.isin(("T3.2",))),
    "interaction"
] = 1

matrix.loc[
    (matrix.Component == "C7") &
    (matrix.Task.isin(("T1.3",))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == "C8") &
    (matrix.Task.isin(("T1.8",))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == "C9") &
    (matrix.Task.isin(("T1.2", "T1.6", "T1.7"))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == "C10") &
    (matrix.Task.isin(("T1.4",))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == "C11") &
    (matrix.Task.isin(("T1.4",))),
    "interaction"
] = 1
matrix.loc[
    (matrix.Component == "C12") &
    (matrix.Task.isin(("T1.9",))),
    "interaction"
] = 1

print(matrix.head())
