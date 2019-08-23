import threading
import time

from tables import *
import tables
import os
from tqdm import tqdm
from data_generation.dimensionality_reduction import DimensionalityReductionKernel
from data_generation.dimensionality_reduction.hdf5_descriptions import TSNEDescription
from utils import Utils


class PersistenceThread(threading.Thread):
    """
    Checks whether threads calculating t-SNE generated new output. If so, new output is stored in file.
    """

    def __init__(
            self,
            results: list,
            expected_number_of_results: int,
            total_number_of_results: int,
            dataset_name: str,
            dim_red_kernel_name: str,
            checking_interval: int = 20
    ):
        """
        Initializes thread for ensuring persistence of t-SNE results calculated by other threads.
        :param results: List of calculated results.
        :param expected_number_of_results: Expected number of datasets to be produced.
        :param total_number_of_results: Number of all results, including already generated ones.
        :param dataset_name: Suffix of dataset to be created.
        :param dim_red_kernel_name: Name of dimensionality reduction kernel used.
        :param checking_interval: Intervals in seconds in which thread checks for new models.
        """
        threading.Thread.__init__(self)

        self._results = results
        self._expected_number_of_results = expected_number_of_results
        self._total_number_of_results = total_number_of_results
        self._dataset_name = dataset_name
        self._dim_red_kernel_name = dim_red_kernel_name
        self._checking_interval = checking_interval
        self._ids_to_process = None
        self._storage_path = os.getcwd() + "/../../data/"

        # Fetch .h5 file handle.
        self._h5file = self._open_pytables_file()

        # Display progress bar.
        self._progress_bar = tqdm(
            total=total_number_of_results, initial=total_number_of_results - expected_number_of_results
        )

    def run(self):
        """
        Persistence thread starts to monitor shared list object holding results. If new elements are added, changes are
        pushed to disk.
        :return:
        """

        metadata_table = self._h5file.root.metadata
        metadata_row = metadata_table.row
        # Get configuration of this DR kernel's parameter set.
        parameter_config = DimensionalityReductionKernel.DIM_RED_KERNELS[self._dim_red_kernel_name]["parameters"]

        # Check on new arrivals every self._checking_interval seconds.
        last_processed_index = -1
        num_of_results_so_far = 0
        while num_of_results_so_far < self._expected_number_of_results:
            # Check how many results are available so far.
            num_of_results_so_far = len(self._results)

            # If number of records has changed: Add to file.
            if num_of_results_so_far > last_processed_index + 1:
                # If new records are available since last loop iteration: Update progress bar.
                self._progress_bar.update(num_of_results_so_far - last_processed_index - 1)

                # Loop through all entries.
                for result in self._results[last_processed_index + 1:num_of_results_so_far]:
                    ######################################################
                    # 1. Add metadata (hyperparameter + objectives).
                    ######################################################

                    # Fetch next viable model ID.
                    valid_model_id = self._ids_to_process.pop()

                    # Generic metadata.
                    metadata_row["id"] = valid_model_id
                    metadata_row["num_records"] = result["low_dimensional_projection"].shape[0]

                    # Hyperparameter.
                    result_hyperparam = result["parameter_set"]

                    # Add hyperparameter values.
                    for param_config in parameter_config:
                        metadata_row[param_config["name"]] = result_hyperparam[param_config["name"]]

                    # Objectives.
                    result_objectives = result["objectives"]
                    metadata_row["runtime"] = result_objectives["runtime"]
                    metadata_row["r_nx"] = result_objectives["r_nx"]
                    metadata_row["b_nx"] = result_objectives["b_nx"]
                    metadata_row["stress"] = result_objectives["stress"]
                    metadata_row["classification_accuracy"] = result_objectives["classification_accuracy"]
                    metadata_row["separability_metric"] = result_objectives["separability_metric"]

                    # Append row to file.
                    metadata_row.append()

                    ######################################################
                    # 2. Add low-dimensional projections.
                    ######################################################

                    self._h5file.create_carray(
                        self._h5file.root.projection_coordinates,
                        name="model" + str(valid_model_id),
                        obj=result["low_dimensional_projection"],
                        title="Low dimensional coordinates for model #" + str(valid_model_id),
                        filters=Filters(complevel=3, complib='zlib')
                    )

                    ######################################################
                    # 3. Add pointwise quality critera values.
                    ######################################################

                    self._h5file.create_carray(
                        self._h5file.root.pointwise_quality,
                        name="model" + str(valid_model_id),
                        obj=result_objectives["pointwise_quality_values"],
                        title="Pointwise quality values for model #" + str(valid_model_id),
                        filters=Filters(complevel=3, complib='zlib')
                    )

                # Flush buffer, make sure data is stored in file.
                metadata_table.flush()

                # Keep track of which records are already saved to disk.
                last_processed_index = num_of_results_so_far - 1

            # Wait a few seconds before checking whether any new elements have been added.
            time.sleep(self._checking_interval)

        # Wrap up progress bar display.
        self._progress_bar.close()

        # Close file.
        print(self._h5file)
        self._h5file.close()

    def _open_pytables_file(self) -> tables.file.File:
        """
        Creates new pytables/.h5 file for dataset with specified name.
        :return: File handle of newly created .h5 file.
        """

        self._ids_to_process = {i for i in range(0, self._total_number_of_results)}
        file_name = self._storage_path + "drop_" + self._dataset_name + "_" + self._dim_red_kernel_name.lower() + ".h5"

        # If file exists: Return handle to existing file (assuming file is not corrupt).
        if os.path.isfile(file_name):
            h5file = open_file(filename=file_name, mode="r+")
            # Determine highest ID of available nodes.
            for low_dim_leaf in h5file.walk_nodes("/projection_coordinates/", classname="CArray"):
                self._ids_to_process.remove(int(low_dim_leaf._v_name[5:]))

            return h5file

        # If file doesn't exist yet: Initialize new file.
        h5file = open_file(filename=file_name, mode="w")

        # Create groups in new file (embedding coordinates and embedding qualities of each point).
        h5file.create_group(h5file.root, "projection_coordinates", title="Low-dimensional coordinates")
        h5file.create_group(h5file.root, "pointwise_quality", title="Pointwise embedding quality")

        # Create table.
        metadata_table = h5file.create_table(
            where=h5file.root,
            name='metadata',
            description=DimensionalityReductionKernel.DIM_RED_KERNELS[self._dim_red_kernel_name]["hdf5_description"],
            title="Metadata for t-SNE models"
        )
        metadata_table.flush()

        return h5file
