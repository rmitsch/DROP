#!/bin/sh

## Manual setup steps. ##

# Download MNIST dataset.
python -c 'from sklearn.datasets import fetch_mldata; fetch_mldata("MNIST original")'

# Download repository and install package for multi-threaded t-SNE.
pip install --no-cache-dir git+https://github.com/DmitryUlyanov/Multicore-TSNE.git
pip install --no-cache-dir git+https://github.com/samueljackson92/coranking.git

# Install fastText with Python bindings for classification of VIS documents.
git clone https://github.com/facebookresearch/fastText.git
cd fastText
python setup.py install