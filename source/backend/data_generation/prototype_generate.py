import sys
from random import shuffle
import psutil
import os
from tables import *
from tqdm import tqdm

import backend.objectives.topology_preservation_objectives.CorankingMatrix as CorankingMatrix
from backend.data_generation.PersistenceThread import PersistenceThread
from backend.data_generation.datasets import *
from backend.data_generation.dimensionality_reduction import DimensionalityReductionKernel
from backend.data_generation.dimensionality_reduction.DimensionalityReductionThread import DimensionalityReductionThread
from backend.utils import Utils


def generate_instance(instance_dataset_name: str):
    """
    Generates and returns dataset instance of specified type.
    :param instance_dataset_name:
    :return:
    """
    if instance_dataset_name not in ("wine", "swiss_roll", "mnist", "vis"):
        raise ValueError('Dataset ' + instance_dataset_name + ' not supported.')

    if instance_dataset_name == "wine":
        return WineDataset()
    elif instance_dataset_name == "swiss_roll":
        return SwissRollDataset()
    elif instance_dataset_name == "mnist":
        return MNISTDataset()
    elif instance_dataset_name == "vis":
        return VISPaperDataset()

# Create logger.
logger = Utils.create_logger()

######################################################
# 1. Generate parameter sets, store in file.
######################################################

# Define name of dataset to use (appended to file name).
dataset_name = sys.argv[1] if len(sys.argv) > 1 else "wine"
# Define DR method to use.
dim_red_kernel_name = sys.argv[2] if len(sys.argv) > 2 else "TSNE"

# Get all parameter configurations (to avoid duplicate model generations).
parameter_sets = DimensionalityReductionKernel.generate_parameter_sets_for_testing(
    data_file_path=os.getcwd() + "/../data/drop_" + dataset_name + "_" + dim_red_kernel_name.lower() + ".h5",
    dim_red_kernel_name=dim_red_kernel_name
)

######################################################
# 2. Load high-dimensional data.
######################################################

logger.info("Creating dataset.")

# Load dataset.
high_dim_dataset = generate_instance(instance_dataset_name=dataset_name)
# Persist dataset's records.
high_dim_dataset.persist_records(directory=os.getcwd() + "/../data")

# Scale attributes, fetch predictors.
high_dim_features = high_dim_dataset.preprocessed_features()

######################################################
# 3. Calculate distance matrices and the corresponding
# coranking matrices.
######################################################

logger.info("Calculating distance matrices.")
distance_matrices = {
    metric: high_dim_dataset.compute_distance_matrix(metric=metric)
    for metric in DimensionalityReductionKernel.get_metric_values(dim_red_kernel_name)
}

# Generate neighbourhood ranking for high dimensional data w.r.t. all used distance metrics.
logger.info("Generating neighbourhood rankings.")
high_dim_neighbourhood_rankings = {
    metric: CorankingMatrix.generate_neighbourhood_ranking(
        distance_matrix=distance_matrices[metric]
    )
    for metric in DimensionalityReductionKernel.get_metric_values(dim_red_kernel_name)
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
logger.info("Generating dimensionality reduction models with " + str(n_jobs) + " threads.")
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
            dim_red_kernel_name=dim_red_kernel_name
        )
    )

# Create thread ensuring persistence of results.
threads.append(
    PersistenceThread(
        results=results,
        expected_number_of_results=len(parameter_sets),
        dataset_name=dataset_name,
        dim_red_kernel_name=dim_red_kernel_name,
        checking_interval=10
    )
)

######################################################
# 3. Calculate low-dim. represenatations.
######################################################

for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
