import os
import csv
import pandas
import nltk
from pattern3 import vector as pattern_vector
import tempfile
import re
import numpy
from fastText import train_supervised
from backend.data_generation import InputDataset


class VISPaperDataset(InputDataset):
    """
    Generates dataset of VIS papers using fastText for generating word embeddings and classification.
    """

    def __init__(self):
        super().__init__()

        self._model = None

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
        data = self._data["features"]

        # Append new column holding the ready-to-train text.
        data["assembled_text"] = ""

        ####################################
        # 1. Preprocess paper text.
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

        # Update value of labels.
        self._data["labels"] = data["cluster_title"]

        ####################################
        # 2. Build fastText model.
        ####################################

        self._model = self._build_fasttext_model()

        return data

    def _build_fasttext_model(self):
        """
        Build fastText model based on available data.
        :return:
        """
        # Create temporary file so fastText reads the data.
        with tempfile.NamedTemporaryFile(mode="wt") as temp:
            # Dump content to file.
            for item in self._data["features"]["assembled_text"].tolist():
                temp.write("%s\n" % item)
            # Rewind temp. file before reading from it.
            temp.seek(0)

            # Train fastText model.
            return train_supervised(
                input=temp.name,
                epoch=100,
                dim=150,
                ws=10,
                lr=0.25,
                wordNgrams=2,
                verbose=2,
                minCount=1
            )

    def calculate_classification_accuracy(self, features: numpy.ndarray = None):
        print("calculating accuracy")
        # Set features, if not specified in function call.
        features = self.preprocessed_features() if features is None else features
        labels = self.labels()

        # Split data. Use several iterations.



        return 0