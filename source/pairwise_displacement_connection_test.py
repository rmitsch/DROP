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

max_cell_idx: int = pairwise_displacement_data.high_dim_neighbour_rank.max()
bin_interval: int = int(max_cell_idx / 15)
for hd_i, hd_idx in enumerate(range(0, max_cell_idx, bin_interval)):
    print(hd_i)
    hd_cell_idxs: set = set(range(hd_idx, hd_idx + bin_interval))

    for ld_i, ld_idx in enumerate(range(0, max_cell_idx, bin_interval)):
        ld_cell_idxs: set = set(range(ld_idx, ld_idx + bin_interval))

        relevant_connections: pd.DataFrame = pairwise_displacement_data[
            pairwise_displacement_data.high_dim_neighbour_rank.isin(hd_cell_idxs) &
            pairwise_displacement_data.low_dim_neighbour_rank.isin(ld_cell_idxs)
        ]

        fig = px.scatter(x=original_dataset[0], y=original_dataset[1])
        for idx, record in relevant_connections.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[record.source_0, record.neighbour_0],
                    y=[record.source_1, record.neighbour_1],
                    mode='lines',
                    name='lines'
                )
            )
        fig.update_layout(title=str(hd_i) + " : " + str(ld_i) + "(" + str(len(relevant_connections)) + ")")
        fig.write_image("/home/raphael/Development/data/DROP/corm-demo-images/" + str(hd_i) + "_" + str(ld_i) + ".png")
