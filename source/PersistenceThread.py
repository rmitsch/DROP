import threading
import time
from tables import *
import TSNEMetadata
import os


class PersistenceThread(threading.Thread):
    """
    Checks whether threads calculating t-SNE generated new output. If so, new output is stored in file.
    """

    def __init__(self, results: list, expected_number_of_results: int, dataset_name: str):
        """
        Initializes thread for ensuring persistence of t-SNE results calculated by other threads.
        :param results: List of calculated results.
        :param expected_number_of_results: Expected number of datasets to be produced.
        :param dataset_name: Suffix of dataset to be created.
        """
        threading.Thread.__init__(self)

        self.results = results
        self.expected_number_of_results = expected_number_of_results
        self.dataset_name = dataset_name

        # Fetch .h5 file handle.
        self.h5file = self._open_pytables_file()

    def run(self):
        """
        Persistence thread starts to monitor shared list object holding results. If new elements are added
        :return:
        """

        metadata_table = self.h5file.root.metadata
        metadata_row = metadata_table.row

        # Check on new arrivals every 5 seconds.
        last_processed_index = -1
        num_of_results_so_far = 0
        while num_of_results_so_far < self.expected_number_of_results:
            # Check how many results are available so far.
            num_of_results_so_far = len(self.results)
            print(num_of_results_so_far / float(self.expected_number_of_results))
            # If number of records has changed: Add to file.
            if num_of_results_so_far > last_processed_index + 1:
                # Loop through all entries.
                for result in self.results[last_processed_index + 1:num_of_results_so_far]:
                    ######################################################
                    # 1. Add metadata (hyperparameter + objectives).
                    ######################################################

                    # Calculate ID to persist based on running ID as assigned by generation procedure plus offset
                    # necessary due to datasets already existing in target file for dataset.
                    valid_model_id = self.model_id_offset + result["parameter_set"]["id"]

                    # Hyperparameter.
                    result_hyperparam = result["parameter_set"]
                    metadata_row["id"] = valid_model_id
                    metadata_row["n_components"] = result_hyperparam["n_components"]
                    metadata_row["perplexity"] = result_hyperparam["perplexity"]
                    metadata_row["early_exaggeration"] = result_hyperparam["early_exaggeration"]
                    metadata_row["learning_rate"] = result_hyperparam["learning_rate"]
                    metadata_row["n_iter"] = result_hyperparam["n_iter"]
                    # metadata_row["min_grad_norm"] = result["min_grad_norm"]
                    metadata_row["angle"] = result_hyperparam["angle"]
                    metadata_row["metric"] = result_hyperparam["metric"]

                    # Objectives.
                    result_objectives = result["objectives"]
                    metadata_row["runtime"] = result_objectives["runtime"]
                    metadata_row["trustworthiness"] = result_objectives["trustworthiness"]

                    # Append row to file.
                    metadata_row.append()

                    ######################################################
                    # 2. Add low-dimensional projections.
                    ######################################################

                    self.h5file.create_carray(
                        self.h5file.root.projection_coordinates,
                        name="model" + str(valid_model_id),
                        obj=result["low_dimensional_projection"],
                        title="Low dimensional coordinates for model #" + str(valid_model_id),
                        filters=Filters(complevel=3, complib='zlib')
                    )

                # Flush buffer, make sure data is stored in file.
                metadata_table.flush()

                # Keep track of which records are already saved to disk.
                last_processed_index = num_of_results_so_far - 1

            # Wait a few seconds before checking whether any new elements have been added.
            time.sleep(5)

        # Close file.
        print(self.h5file)
        self.h5file.close()

    def _open_pytables_file(self):
        """
        Creates new pytables/.h5 file for dataset with specified name.
        :return: File handle of newly created .h5 file.
        """

        # Used to store how many models are already stored in file.
        self.model_id_offset = 0

        file_name = os.getcwd() + "/../data/drop_" + self.dataset_name + ".h5"
        # If file exists: Return handle to existing file (assuming file is not corrupt).
        if os.path.isfile(file_name):
            h5file = open_file(filename=file_name, mode="r+")
            # Determine highest ID of available nodes.
            for low_dim_leaf in h5file.walk_nodes("/projection_coordinates/", classname="CArray"):
                model_id = int(low_dim_leaf._v_name[5:])
                self.model_id_offset = model_id if model_id > self.model_id_offset else self.model_id_offset

            # Set offset to first ID of new model.
            self.model_id_offset += 1

            return h5file

        # If file doesn't exist yet: Initialize new file.
        h5file = open_file(filename=file_name, mode="w")

        # Create groups in new file.
        h5file.create_group(h5file.root, "projection_coordinates", title="Low-dimensional coordinates")

        # Create table.
        metadata_table = h5file.create_table(
            where=h5file.root,
            name='metadata',
            description=TSNEMetadata.TSNEMetadata,
            title="Metadata for t-SNE models"
        )
        metadata_table.flush()

        return h5file
