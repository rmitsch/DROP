from random import shuffle
import os

import psutil
from tables import *

import backend.objectives.topology_preservation_objectives.CorankingMatrix as CorankingMatrix
from backend.data_generation.PersistenceThread import PersistenceThread
from backend.data_generation.TSNEThread import TSNEThread
from backend.data_generation.DimensionaliyReductionThread import DimensionalityReductionThread
from backend.data_generation.datasets.WineDataset import WineDataset
from backend.utils import Utils

# Create logger.
logger = Utils.create_logger()

######################################################
# 1. Generate parameter sets, store in file.
######################################################

# Define name of dataset to use (appended to file name).
dataset_name = "wine"
# Define DR method to use.
dim_red_kernel = "TSNE"

# Get all parameter configurations (to avoid duplicate model generations).
existent_parameter_sets = []
file_name = os.getcwd() + "/../data/drop_" + dataset_name + "_" + dim_red_kernel + ".h5"
if os.path.isfile(file_name):
    h5file = open_file(filename=file_name, mode="r+")
    for row in h5file.root.metadata:
        # Note: We don't use the model ID here, since that would never lead to comparison hits.
        existent_parameter_sets.append({
            "n_components": row["n_components"],
            "perplexity": row["perplexity"],
            "early_exaggeration": row["early_exaggeration"],
            "learning_rate": row["learning_rate"],
            "n_iter": row["n_iter"],
            # "min_grad_norm": row["min_grad_norm"],
            "angle": row["angle"],
            "metric": str(row["metric"], "utf-8")
        })

    # Close file after reading parameter set data.
    h5file.close()


# Define parameter ranges.
parameter_values = {
    "n_components": (1, 2), #2, 3, 4),
    "perplexity": (10, 25), #25, 50, 80),
    "early_exaggeration": (5.0, 10), #10.0, 15.0, 20.0),
    "learning_rate": (10.0, 250), #, 250.0), # 500.0, 1000.0),
    "n_iter": (100, 250, 500), #1000, 2000, 5000),
    # Commenting out min_grad_norm, since a variable value for this since (1) MulticoreTSNE doesn't support dynamic
    # values for this attribute and (2) sklearn's implementation is slow af.
    # If a decently performing implementation (sklearn updates?) that also supports this parameter is ever available,
    # it might be added again.
    #"min_grad_norm": (1e-10, 1e-7, 1e-4, 1e-1),
    "angle": (0.1, 0.35), #0.35, 0.65, 0.9),
    "metrics": ('cosine', 'euclidean') #, 'euclidean')
}

# Filter out already existing model parametrizations.
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
                                new_parameter_set = {
                                    "n_components": n_components,
                                    "perplexity": perplexity,
                                    "early_exaggeration": early_exaggeration,
                                    "learning_rate": learning_rate,
                                    "n_iter": n_iter,
                                    # "min_grad_norm": min_grad_norm,
                                    "angle": angle,
                                    "metric": metric
                                }

                                # If new parameter set not already generated: Add to list of datasets to generate.
                                if new_parameter_set not in existent_parameter_sets:
                                    new_parameter_set["id"] = current_id
                                    parameter_sets.append(new_parameter_set)

                                # Keep track of number of generated parameter sets.
                                current_id += 1

######################################################
# 2. Load high-dimensional data.
######################################################

logger.info("Creating dataset.")

# Load dataset.
high_dim_dataset = WineDataset()

# Scale attributes, fetch predictors.
high_dim_features = high_dim_dataset.preprocessed_features()

######################################################
# 3. Calculate distance matrices and the corresponding
# coranking matrices.
######################################################

logger.info("Calculating distance matrices.")
distance_matrices = {
    metric: high_dim_dataset.compute_distance_matrix(metric=metric)
    for metric in parameter_values["metrics"]
}

# Generate neighbourhood ranking for high dimensional data w.r.t. all used distance metrics.
logger.info("Generating neighbourhood rankings.")
high_dim_neighbourhood_rankings = {
    metric: CorankingMatrix.generate_neighbourhood_ranking(distance_matrix=distance_matrices[metric])
    for metric in parameter_values["metrics"]
}

######################################################
# 3. Set up multithreading.
######################################################

# Shuffle list with parameter sets so that they are kinda evenly distributed.
shuffle(parameter_sets)
# Determine number of workers.
n_jobs = psutil.cpu_count(logical=True) - 1
threads = []
# Shared list holding results.
results = []

# Split parameter sets amongst workers.
logger.info("Generating t-SNE models.")
num_parameter_sets = int(len(parameter_sets) / n_jobs)
for i in range(0, n_jobs):
    first_index = num_parameter_sets * i
    # Add normal number of parameter sets, if this isn't the last job. Otherwise add all remaining sets.
    last_index = first_index + num_parameter_sets if i < (n_jobs - 1) else len(parameter_sets)

    # Instantiate thread.
    threads.append(
        DimensionalityReductionThread(
            results=results,
            distance_matrices=distance_matrices,
            parameter_sets=parameter_sets[first_index:last_index],
            input_dataset=high_dim_dataset,
            high_dimensional_neighbourhood_rankings=high_dim_neighbourhood_rankings,
            dim_red_kernel=dim_red_kernel
        )
    )

# Create thread ensuring persistence of results.
threads.append(
    PersistenceThread(
        results=results,
        expected_number_of_results=len(parameter_sets),
        dataset_name=dataset_name
    )
)

######################################################
# 3. Calculate low-dim. represenatations.
######################################################

print("Generating models.")
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
