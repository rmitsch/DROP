"""
Parameters to use for run.
Note: While this _should_ be identical with values in currently used .hdf5 dump, frontend doesn't trust information from
backend on this and hence runs over dataset once to identify unique values. This doesn't impose a performance penalty
(runs only for 20ms) and makes data handling more robust, especially with varying result set for same dataset.
"""

DIM_RED_KERNELS_PARAMETERS = {
    "TSNE": [
        #     {"name": "n_components", "type": "numeric", "values": [1, 2, 3]}, # [1, 2, 3, 4]},
        #     {"name": "perplexity", "type": "numeric", "values": [10, 25, 50]}, # [10, 25, 50, 100]},
        #     {"name": "early_exaggeration", "type": "numeric", "values": [5, 10, 15]}, # [5, 10, 15, 20]},
        #     {"name": "learning_rate", "type": "numeric", "values": [10, 100, 250]}, # [10, 50, 100, 250, 500]},
        #     {"name": "n_iter", "type": "numeric", "values": [100, 250, 500]}, # [100, 250, 500, 1000]},
        #     {"name": "angle", "type": "numeric", "values": [0.35, 0.5, 0.75]}, # [0.1, 0.35, 0.5, 0.75, 0.9]},
        #     {"name": "metric", "type": "categorical", "values": ['cosine', 'euclidean']}
        {"name": "n_components", "type": "numeric", "values": [1, 2, 3, 4]},
        {"name": "perplexity", "type": "numeric", "values": [10, 25, 50, 100]},
        {"name": "early_exaggeration", "type": "numeric", "values": [5, 10, 15, 20]},
        {"name": "learning_rate", "type": "numeric", "values": [10, 100, 250, 500]},
        {"name": "n_iter", "type": "numeric", "values": [100, 250, 500, 1000]},
        {"name": "angle", "type": "numeric", "values": [0.35, 0.5, 0.75, 0.9]},
        {"name": "metric", "type": "categorical", "values": ['cosine', 'euclidean']}
    ],
    "SVD": [
        {"name": "n_components", "type": "numeric", "values": [1, 2]},
        {"name": "n_iter", "type": "numeric", "values": [5, 10, 20]},
        {"name": "metric", "type": "categorical", "values": ['cosine', 'euclidean']}
    ],
    "UMAP": [
        {"name": "n_components", "type": "numeric", "values": [2, 3, 4]},
        {"name": "n_neighbors", "type": "numeric", "values": [4, 10, 15]},
        {"name": "n_epochs", "type": "numeric", "values": [200, 500, 750]},
        {"name": "learning_rate", "type": "numeric", "values": [0.1, 0.5, 1.0]},
        {"name": "min_dist", "type": "numeric", "values": [0.05, 0.1, 0.2]},
        {"name": "local_connectivity", "type": "numeric", "values": [1, 2, 3]},
        {"name": "metric", "type": "categorical", "values": ['euclidean', 'cosine']}
    ]
}