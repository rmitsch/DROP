import logging
from contextlib import contextmanager
import sys
import os
from sklearn.tree import DecisionTreeRegressor


# Class for various, non-essential tasks.
class Utils:
    # Inactive constructor.
    def __init__(self):
        print("Constructing Utils. Shouldn't happen")

    # Construct logging object.
    @staticmethod
    def create_logger():
        # Logger set-up.
        # Source: https://docs.python.org/3/howto/logging-cookbook.html
        # Create global logger.
        Utils.logger = logging.getLogger('DROP')
        Utils.logger.setLevel(logging.DEBUG)
        # Create console handler.
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # Create formatter and add it to handlers.
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s | %(message)s')
        ch.setFormatter(formatter)
        # Add handlers to logger.
        Utils.logger.addHandler(ch)

        # Return logger object.
        return Utils.logger

    @staticmethod
    def extract_decision_tree_structure(clf: DecisionTreeRegressor, features: list, labels: list, node_index: int = 0):
        """
        Return textual structure for generated decision tree.
        Source: https://planspace.org/20151129-see_sklearn_trees_with_d3/.
        :return: Textual representation of regression tree.
        """

        node = {}

        if clf.tree_.children_left[node_index] == -1:  # indicates leaf
            node['name'] = " | ".join([
                                          label + ": " +
                                          str(round(clf.tree_.value[node_index][i][0], 3))
                                          for i, label in enumerate(labels[0])
                                          ])

        else:
            feature = features[clf.tree_.feature[node_index]]
            threshold = clf.tree_.threshold[node_index]
            node['name'] = '{} > {}'.format(feature, round(threshold, 3))
            left_index = clf.tree_.children_left[node_index]
            right_index = clf.tree_.children_right[node_index]

            node['children'] = [
                Utils.extract_decision_tree_structure(clf, features, labels, right_index),
                Utils.extract_decision_tree_structure(clf, features, labels, left_index)
            ]

        return node

    @staticmethod
    @contextmanager
    def suppress_stdout():
        """
        Surpresses output by creating a context object.
        Source: https://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/
        :return:
        """
        with open(os.devnull, "w") as devnull:
            old_stderr = sys.stderr
            old_stdout = sys.stdout
            sys.stdout = devnull
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
