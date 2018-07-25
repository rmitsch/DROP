#!/bin/sh

## Manual setup steps. ##

# Note: All Numba decorators in umap_.py have to be set to @numba.njit(parallel=False, fastmath=True)!
# Otherwise external thread-level parallelism leads to deadlocking threads due to Numba parallelization.
# todo Change numba decorators with search/replace in file.

# Download MNIST dataset.
python -c 'from sklearn.datasets import fetch_mldata; fetch_mldata("MNIST original")'

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