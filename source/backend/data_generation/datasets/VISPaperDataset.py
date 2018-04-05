import pandas
import nltk
from pattern3 import vector as pattern_vector
import tempfile
import re
import numpy
import fastText
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist

from backend.data_generation import InputDataset


class VISPaperDataset(InputDataset):
    """
    Generates dataset of VIS papers using fastText for generating word embeddings and classification.
    """

    def __init__(self):
        # Holding fastText model.
        self._model = None

        super().__init__()

    def labels(self):
        return self._data["labels"]

    def features(self):
        return self._data["features"]

    def _load_data(self):
        data = {
            # Load paper records as dataframe.
            "features": pandas.read_csv(
                filepath_or_buffer='../data/vis_papers/vis_papers.csv',
                delimiter=',',
                quotechar='"'
            ),
            "labels": None
        }

        # Preprocess data.
        data["labels"] = data["features"]["keywords"].str.split(",")

        return data

    def _preprocess_features(self):
        """
        Preprocesses VIS paper texts.
        :return:
        """

        ####################################
        # Preprocess paper texts.
        ####################################

        # 1. Define stopwords.
        stopwords = nltk.corpus.stopwords.words('english')
        stopwords.extend([
            "visualization", "paper", "we", "new", "recent", "using", "apply", "study", "many", "areas", "common",
            "goal", "provided", "existing", "arise", "data", "sup", "used", "whose", "via", "main", "yet", "proven",
            "given", "visualizing", "many", "newly", "developed", "address", "problem", "show", "introduce", "present",
            "novel", "analyzing", "present", "every", "also", "within", "visual", "studies", "visualanalytics", "us",
            "becomes", "take", "advantage", "understand", "better", "much", "describe", "support", "generally",
            "previous", "work", "basis", "true", "contain", "serves", "approach", "promising"
        ])

        # 2. Drop all labels below defined occurence threshold.
        infrequent_labels = self._select_infrequent_labels(threshold=5)
        filtered_labels = [None] * len(self._data["labels"])
        # Loop through all records.
        for i in range(0, len(self._data["labels"])):
            filtered_labels[i] = []
            # Loop through all labels in this record.
            for label in self._data["labels"][i]:
                # Add to new label list, if not infrequent.
                if label not in infrequent_labels:
                    filtered_labels[i].append(label.strip())

        # Update labels.
        self._data["labels"] = filtered_labels
        self._data["features"]["keywords"] = filtered_labels

        # 3. Drop all papers having no labels after removal of infrequent ones.
        features = self._data["features"]
        self._data["features"] = features[features["keywords"].apply(len) > 0]
        self._data["features"] = self._data["features"].reset_index(drop=True)
        self._data["labels"] = self._data["features"]["keywords"]

        # 4. Preprocess text - apply modification to all rows.
        self._data["features"] = self._preprocess_paper_records(stopwords=stopwords)

        return self._data["features"]

    def _select_infrequent_labels(self, threshold: int = 1):
        """
        Selects titles of infrequent labels.
        Necessary since we can't create models with only one member per class.
        :param threshold: Maximum number of occurences to count as infrequent label.
        :return: Set containing titles of infrequent labels.
        """

        features = self._data["features"]
        counts_per_label = {}
        for i in features.index:
            # Go through all labels.
            for label in self._data["labels"][i]:
                if label not in counts_per_label:
                    counts_per_label[label] = 1
                else:
                    counts_per_label[label] += 1

        cluster_memberships = {k: v for k, v in counts_per_label.items() if v < threshold}

        return set(cluster_memberships.keys())

    def _preprocess_paper_records(self, stopwords: list):
        """
        Preprocesses all records in specified dataframe.
        :param stopwords:
        :return: Preprocessed dataframe.
        """

        data = self._data["features"]

        # Append new column holding the ready-to-train text.
        data["assembled_text"] = ""
        data["assembled_text_wo_labels"] = ""

        # Preprocess text - apply modification to all rows.
        for i in data.index:
            # Remove special characters.
            processed_abstract = re.sub(
                r"([;]|[(]|[)]|[/]|[\\]|[$]|[&]|[>]|[<]|[=]|[:]|[,]|[.]|[-]|(\d+)|[']|[\"])",
                "",
                data.ix[i, 'abstract']
            )
            processed_title = re.sub(
                r"([;]|[(]|[)]|[/]|[\\]|[$]|[&]|[>]|[<]|[=]|[:]|[,]|[.]|[-]|(\d+)|[']|[\"])",
                "",
                data.ix[i, 'title']
            )

            # Lemmatize text.
            processed_abstract = [
                pattern_vector.stem(word, stemmer=pattern_vector.LEMMA)
                for word in processed_abstract.split()
            ]
            processed_title = [
                pattern_vector.stem(word, stemmer=pattern_vector.LEMMA)
                for word in processed_title.split()
            ]

            # Remove stopwords.
            data.ix[i, 'abstract'] = ' '.join(filter(lambda x: x not in stopwords, processed_abstract))
            data.ix[i, 'title'] = ' '.join(filter(lambda x: x not in stopwords, processed_title))

            # Concatenate cluster titles to single term.
            data.ix[i, 'cluster_title'] = re.sub(
                r"([,]|[/]|[&]|[-]|[+]|[(]|[)])",
                "_",
                data.ix[i, 'cluster_title'].replace(" ", "").replace("'", "")
            )

            # Prepare prefixes for label.
            keyword_label_string = ""
            for label in self._data["labels"][i]:
                keyword_label_string += \
                    "__label__" + \
                    label.replace(" ", "_") + \
                    label.replace("/", "_") + \
                    label.replace("&", "_") + \
                    " "

            # Assemble preprocessed abstract, title and labels.
            # "__label__" + data.ix[i, 'cluster_title'] + " " + \
            data.ix[i, 'assembled_text'] = \
                keyword_label_string.strip() + \
                data.ix[i, 'title'] + " " + \
                data.ix[i, 'abstract']
            # Assemble preprocessed abstract, title and labels w/o cluster title.
            data.ix[i, 'assembled_text_wo_labels'] = \
                str(self._data["labels"][i]) + " " + \
                data.ix[i, 'title'] + " " + \
                data.ix[i, 'abstract']

        return data

    def _build_fasttext_model(self, indices: numpy.ndarray):
        """
        Build fastText model based on available data.
        :param indices: Indices in self._data["features"] to use for building the model.
        :return:
        """
        # Create temporary file so fastText can read the data.
        with tempfile.NamedTemporaryFile(mode="wt") as temp:
            # Dump content to file.
            for item in self._data["features"]["assembled_text"].values[indices]:
                temp.write("%s\n" % item)
            # Rewind temp. file before reading from it.
            temp.seek(0)

            # Train fastText model.
            return fastText.train_supervised(
                input=temp.name,
                epoch=300,
                dim=150,
                ws=5,
                lr=0.3,
                wordNgrams=2,
                verbose=0,
                minCount=1,
                loss="hs"
            )

    def _evalute_fasttext_model(self, indices: numpy.ndarray, k: int=1):
        """
        Evaluate fastText model based on available data.
        :param indices: Indices in self._data["features"] to use for evaluating the model.
        :param k: Number of most probable labels to consider in test.
        :return:
        """

        # Create temporary file so fastText can read the data.
        with tempfile.NamedTemporaryFile(mode="wt") as temp:
            # Dump content to file.
            for item in self._data["features"]["assembled_text"].values[indices]:
                temp.write("%s\n" % item)
            # Rewind temp. file before reading from it.
            temp.seek(0)

            # Test fastText model.
            return self._model.test(
                path=temp.name,
                k=k
            )

    def calculate_classification_accuracy(self, features: numpy.ndarray = None):
        # ********** TODOS **********
        #     - Datasets
        #         * VIS papers
        #               + solve distance matrix organization - either let inputdataset calc. it or provided a numeric
        #                 represenation of word vectors (after model generation?).
        #               + decide & implement approach for class. accuracy of low-dim. projections - still using fasttext class.?
        #                 usual random forest instead? if ft: how to replace original with low-dim. vectors?
        #                 conclusio: random forest for low-dim. probably more reasonable.
        self._logger.info("Calculating classification accuracy.")

        accuracy = 0
        n_splits = 3

        # If this is the original dataset: Classify with fastText using sentence embeddings.
        if features is None:
            # Set features, if not specified in function call.
            features = self.preprocessed_features()

            # Loop through stratified splits, average prediction accuracy over all splits.
            for i in range(0, n_splits):
                train_indices, test_indices = train_test_split(
                    numpy.arange(len(features)),
                    test_size=0.33
                )

                self._model = self._build_fasttext_model(train_indices)
                ft_eval = self._evalute_fasttext_model(indices=test_indices, k=5)

                # Calculate accuracy as avg. of precision and recall (crude, I know).
                accuracy += (ft_eval[1] + ft_eval[2]) / 2.0

        # If not: Use multi-label random forest using low-dim. projection.
        else:
            pass

        return accuracy / n_splits

    def compute_distance_matrix(self, metric: str):
        # 1. Fetch document vectors for papers.
        features = self._data["features"]
        numerical_values = [None] * len(features)
        for i in features.index:
            numerical_values[i] = self._model.get_sentence_vector(features.ix[i, "assembled_text_wo_labels"])

        # 2. Calculate distance matrix.
        return cdist(numpy.asarray(numerical_values), numpy.asarray(numerical_values), metric)
