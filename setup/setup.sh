#!/bin/sh

## Manual setup steps. ##

# Note: All Numba decorators in umap_.py have to be set to @numba.njit(parallel=False, fastmath=True)!
# Otherwise external thread-level parallelism leads to deadlocking threads due to Numba parallelization.
sed -i 's/numba.njit(parallel=True/numba.njit(parallel=False/g' /usr/local/lib/python3.5/site-packages/umap/umap_.py 

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

fetch_mnist(data_home=None)
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