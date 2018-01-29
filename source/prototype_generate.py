from sklearn.datasets import load_wine
from sklearn.manifold import TSNE
import time
import numpy
from sklearn.preprocessing import StandardScaler
from tables import *
import os.path
from random import shuffle
import multiprocessing
from scipy.spatial.distance import cdist
import psutil
from TSNEMetadata import TSNEMetadata
from TSNEThread import TSNEThread
from PersistenceThread import PersistenceThread
import MulticoreTSNE


######################################################
# 1. Generate parameter sets, store in file.
######################################################

# Define parameter ranges.
parameter_values = {
    "n_components": (1, ), #2, 3, 4),
    "perplexity": (10, ), #25, 50, 80),
    "early_exaggeration": (5.0, 10.0, ), #15.0, 20.0),
    "learning_rate": (10.0, 250.0, ), #500.0, 1000.0),
    "n_iter": (250, 1000, 2000, 5000),
    # Commenting out min_grad_norm, since a variable value for this since (1) MulticoreTSNE doesn't support dynamic
    # values for this attribute and (2) sklearn's implementation is slow af.
    # If a decently performing implementation (sklearn updates?) that also supports this parameter is ever available,
    # it might be added again.
    #"min_grad_norm": (1e-10, 1e-7, 1e-4, 1e-1),
    "angle": (0.1, 0.35, 0.65, 0.9),
    "metrics": ('euclidean',)
}

parameter_sets = []
current_id = 0
for n_components in parameter_values["n_components"]:
    for perplexity in parameter_values["perplexity"]:
        for early_exaggeration in parameter_values["early_exaggeration"]:
            for n_iter in parameter_values["n_iter"]:
                for learning_rate in parameter_values["learning_rate"]:
                    #for min_grad_norm in parameter_values["min_grad_norm"]:
                        for angle in parameter_values["angle"]:
                            for metric in parameter_values["metrics"]:
                                # Define dictionary object with values.
                                parameter_sets.append({
                                    # Use running ID for later reference between models and their output data.
                                    "id": current_id,
                                    "n_components": n_components,
                                    "perplexity": perplexity,
                                    "early_exaggeration": early_exaggeration,
                                    "learning_rate": learning_rate,
                                    "n_iter": n_iter,
                                    # "min_grad_norm": min_grad_norm,
                                    "angle": angle,
                                    "metric": metric
                                })

                                # Keep track of number of generated parameter sets.
                                current_id += 1

######################################################
# 2. Load high-dimensional data.
######################################################

# Load toy example dataset.
high_dim_data = load_wine()
# Scale attributes.
high_dim_data = StandardScaler().fit_transform(high_dim_data.data)

######################################################
# 3. Calculate distance matrices.
######################################################

# Assumes all specified distances can be calculated by scipy's cdist().
distance_matrices = {
    metric: cdist(high_dim_data.data, high_dim_data.data, metric) for metric in parameter_values["metrics"]
}

######################################################
# 3. Set up multithreading.
######################################################

# Shuffle list with parameter sets.
shuffle(parameter_sets)
# Determine number of workers.
n_jobs = psutil.cpu_count(logical=True) - 1
threads = []
# Shared list holding results.
results = []

# Split parameter sets amongst workers.
num_parameter_sets = int(len(parameter_sets) / n_jobs)
for i in range(0, n_jobs):
    first_index = num_parameter_sets * i
    # Add normal number of parameter sets, if this isn't the last job. Otherwise add all remaining sets.
    last_index = first_index + num_parameter_sets if i < (n_jobs - 1) else len(parameter_sets)

    # Instantiate thread.
    threads.append(
        TSNEThread(
            results=results,
            distance_matrices=distance_matrices,
            parameter_sets=parameter_sets[first_index:last_index],
            high_dimensional_data=high_dim_data
        )
    )

# Create thread ensuring persistence of results.
threads.append(PersistenceThread(
    results=results, expected_number_of_results=len(parameter_sets), dataset_name="wine")
)

######################################################
# 3. Calculate low-dim. represenatations.
######################################################

for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
