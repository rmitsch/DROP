import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px

original_dataset: pd.DataFrame = pd.read_pickle("/tmp/original_dataset.pkl").set_index("id")

pairwise_displacement_data: pd.DataFrame = pd.read_pickle(
    "/tmp/pairwise_displacement_data.pkl"
).merge(
    right=original_dataset[[0, 1]], left_on="source", right_on="id"
).rename(
    columns={0: "source_0", 1: "source_1"}
).merge(
    right=original_dataset[[0, 1]], left_on="neighbour", right_on="id"
).rename(
    columns={0: "neighbour_0", 1: "neighbour_1"}
)


with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
    print(pairwise_displacement_data.head())
    print(original_dataset.head())

fig = px.scatter(x=original_dataset[0], y=original_dataset[1])
# todo
#  - evaluate data, pick data fitting "cell".
#  - add line traces for each pair of records in data selected in previous step.
# fig.show()

