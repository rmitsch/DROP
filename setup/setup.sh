#!/bin/sh

## Manual setup steps. ##

# Note: All Numba decorators in umap_.py have to be set to @numba.njit(parallel=False, fastmath=True)!
# Otherwise external thread-level parallelism leads to deadlocking threads due to Numba parallelization.
# todo Change numba decorators with search/replace in file.

# Download MNIST dataset.
# Source: https://github.com/scikit-learn/scikit-learn/issues/8588.
# Workaround for sklearn download problems.
python -c '
from shutil import copyfileobj
from six.moves import urllib
from sklearn.datasets.base import get_data_home
import os

def fetch_mnist(data_home=None):
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    data_home = get_data_home(data_home=data_home)
    data_home = os.path.join(data_home, "mldata")
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    mnist_save_path = os.path.join(data_home, "mnist-original.mat")
    if not os.path.exists(mnist_save_path):
        mnist_url = urllib.request.urlopen(mnist_alternative_url)
        with open(mnist_save_path, "wb") as matlab_file:
            copyfileobj(mnist_url, matlab_file)
'
# NOTE: Code above has to be checked for validity, if sklearn-download can't be fixed.
#python -c 'from sklearn.datasets import fetch_mldata; fetch_mldata("MNIST original")'

# Download repository and install package for multi-threaded t-SNE.
pip install --no-cache-dir git+https://github.com/DmitryUlyanov/Multicore-TSNE.git
pip install --no-cache-dir git+https://github.com/samueljackson92/coranking.git

# Download language data.
python -m spacy download en
# Download NLTK's default stopwords us.
python -c "import nltk; nltk.download('stopwords')"
# Download NLTK tokenizers.
python -c "import nltk; nltk.download('punkt')"

# Install fastText with Python bindings for classification of VIS documents.
git clone https://github.com/facebookresearch/fastText.git
cd fastText
python setup.py install

# Build image (build file to specify instead of .):
# docker build -t "drop-0.4.0" -f Dockerfile .
# Run container:
# docker run --name DROP -t -d -v /home/raphael/Development/data/DROP:/data drop-0.4.0:latest
# Execute data generation script:
# docker exec DROP python /source/backend/data_generation/prototype_generate.py

# docker run -d --name DROP -v /home/raphael/Development/data/DROP:/data drop-0.4.0:latest python /source/backend/data_generation/prototype_generate.py

# Open bin bash interactively:
# docker run -it drop-0.4.0:latest /bin/bash
# Enter /bin/bash for running non-interactive container:
# docker exec -it DROP /bin/bash OR docker attach ID

# Clean-up commands.
# kill all running containers with docker kill $(docker ps -q)
# delete all stopped containers with docker rm $(docker ps -a -q)
# delete all images with docker rmi $(docker images -q)