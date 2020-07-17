import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df: pd.DataFrame = pd.read_csv("/home/raphael/Development/TALE/study_results.csv").head(15)
df = df[["Subject", *[col for col in df.columns if col.startswith("DR") or col.startswith("SUS")]]]
df["DR experience (1-3)"] = df["DR experience (1-3)"].replace("2-3", "2.5").astype(float)
#df["DR experience (1-3)"] = (df["DR experience (1-3)"] + 1) / 4 * 3
print(df["DR experience (1-3)"].mean())
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
    print(df)

sns.boxplot(
    x="Question",
    y="Score",
    data=df.drop(columns=["DR experience (1-3)", "SUS"]).melt(id_vars="Subject", value_name="Score").rename(
        columns={"variable": "Question"}
    ),
    color="blue"
)
plt.show()
sns.boxplot(
    x="Question",
    y="Score",
    data=df[["Subject", "SUS"]].melt(id_vars="Subject", value_name="Score").rename(
        columns={"variable": "Question"}
    ),
    orient="v"
)
plt.show()
