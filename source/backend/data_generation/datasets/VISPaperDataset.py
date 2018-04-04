import pandas
import nltk
from pattern3 import vector as pattern_vector
import tempfile
import re
import numpy
import fastText
from sklearn.model_selection import StratifiedShuffleSplit

from backend.data_generation import InputDataset


class VISPaperDataset(InputDataset):
    """
    Generates dataset of VIS papers using fastText for generating word embeddings and classification.
    """

    def __init__(self):
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
        data["labels"] = data["features"]["cluster_title"]

        return data

    def _preprocess_features(self):
        """
        Preprocesses VIS paper texts.
        :return:
        """

        ####################################
        # Preprocess paper texts.
        ####################################

        # Define stopwords.
        stopwords = nltk.corpus.stopwords.words('english')
        stopwords.extend([
            "visualization", "paper", "we", "new", "recent", "using", "apply", "study", "many", "areas", "common",
            "goal", "provided", "existing", "arise", "data", "sup", "used", "whose", "via", "main", "yet", "proven",
            "given", "visualizing", "many", "newly", "developed", "address", "problem", "show", "introduce", "present",
            "novel", "analyzing", "present", "every", "also", "within"
        ])

        # Preprocess text - apply modification to all rows.
        self._data["features"] = self._preprocess_paper_records(stopwords=stopwords)

        # Drop all records being the only representatives of a cluster.
        self._data["features"] = self._data["features"][
            self._data["features"]['cluster_title'].isin(self._select_nonsingular_clusters())
        ]

        # Update values of labels after preprocessing.
        self._data["labels"] = self._data["features"]["cluster_title"]

        return self._data["features"]

    def _select_nonsingular_clusters(self):
        """
        Selects titles of non-singular (i. e. with more than one entry) clusters.
        Necessary since we can't create models with only one member per class.
        :return: List containing titles of non-singular clusters.
        """

        data = self._data["features"]
        unique, counts = numpy.unique(data["cluster_title"], return_counts=True)
        cluster_memberships = {k: v for k, v in dict(zip(unique, counts)).items() if v > 1}

        return cluster_memberships.keys()

    def _preprocess_paper_records(self, stopwords: list):
        """
        Preprocesses all records in specified dataframe.
        :param stopwords:
        :return: Preprocessed dataframe.
        """

        data = self._data["features"]

        # Append new column holding the ready-to-train text.
        data["assembled_text"] = ""
        data["assembled_text_wo_cluster_title"] = ""

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

            # Replace "," in keywords with space.
            # Note: Keywords could also be used as labels. Instead only cluster is used; keywords are appended to
            # title/abstract.
            keywords = data.ix[i, 'keywords'].replace(",", " ")

            # Concatenate cluster titles to single term.
            data.ix[i, 'cluster_title'] = re.sub(
                r"([,]|[/]|[&]|[-]|[+]|[(]|[)])",
                "_",
                data.ix[i, 'cluster_title'].replace(" ", "").replace("'", "")
            )

            # Assemble preprocessed abstract, title and labels.
            data.ix[i, 'assembled_text'] = \
                "__label__" + data.ix[i, 'cluster_title'] + " " + \
                data.ix[i, 'title'] + " " + \
                keywords + " " + \
                data.ix[i, 'abstract']
            # Assemble preprocessed abstract, title and labels w/o cluster title.
            data.ix[i, 'assembled_text_wo_cluster_title'] = \
                data.ix[i, 'title'] + " " + \
                keywords + " " + \
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
                epoch=100,
                dim=150,
                ws=10,
                lr=0.25,
                wordNgrams=2,
                verbose=2,
                minCount=1
            )

    def _evalute_fasttext_model(self, fasttext_model: fastText.FastText._FastText, indices: numpy.ndarray):
        """
        Evaluate fastText model based on available data.
        :param fasttext_model:
        :param indices: Indices in self._data["features"] to use for evaluating the model.
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
            return fasttext_model.test(
                path=temp.name,
                k=1
            )

    def calculate_classification_accuracy(self, features: numpy.ndarray = None):
        # ********** TODOS **********
        #     - Datasets
        #         * VIS papers
        #               + calc. class. accuracy.
        #               + solve distance matrix organization - either let inputdataset calc. it or provided a numeric
        #                 represenation of word vectors (after model generation?).
        #               + decide & implement approach for class. accuracy of low-dim. projections - still using fasttext class.?
        #                 usual random forest instead? if ft: how to replace original with low-dim. vectors?
        #                 conclusio: random forest for low-dim. probably more reasonable.

        # Get labels.
        labels = self.labels()

        # If this is the original dataset: Classify using fastText.
        if features is None:
            print("calculating accuracy")

            # Set features, if not specified in function call.
            features = self.preprocessed_features()

            # Loop through stratified splits, average prediction accuracy over all splits.
            accuracy = 0
            n_splits = 3
            for train_indices, test_indices in StratifiedShuffleSplit(
                    n_splits=n_splits,
                    test_size=0.33
            ).split(features["assembled_text_wo_cluster_title"].values, labels.values):
                model = self._build_fasttext_model(train_indices)
                eval = self._evalute_fasttext_model(fasttext_model=model, indices=test_indices)
                print(eval)


        exit()
        return 0
