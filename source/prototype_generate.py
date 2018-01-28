from sklearn.datasets import load_wine
from sklearn.manifold import TSNE
import time
import numpy
from sklearn.preprocessing import StandardScaler
from tables import *

# Load toy example dataset.
high_dim_data = load_wine()
# Scale attributes.
high_dim_data = StandardScaler().fit_transform(high_dim_data.data)

# Create new file.
print(time.time())

# Define parameter ranges.
parameter_values = {
    "n_components": (1, 2, 3, 4, 5, 6, 7),
    "perplexity": (10, 25, 50, 70, 90),
    "early_exaggeration": (1.0, 5.0, 10.0, 15.0, 20.0),
    "learning_rate": (10.0, 100.0, 250.0, 500.0, 1000.0),
    "n_iter": (200, 500, 1000, 2000, 5000),
    "min_grad_norm": (1e-10, 1e-8, 1e-6, 1e-4, 1e-2),
    "angle": (0.1, 0.25, 0.5, 0.75, 0.9)
}

count = 0
for n_components in parameter_values["n_components"]:
    for perplexity in parameter_values["perplexity"]:
        for early_exaggeration in parameter_values["early_exaggeration"]:
            for n_iter in parameter_values["n_iter"]:
                for min_grad_norm in parameter_values["min_grad_norm"]:
                    for angle in parameter_values["angle"]:
                        count += 1

print(count)
start = time.time()
X_embedded = TSNE(n_components=5, method="exact").fit_transform(high_dim_data.data)
print(time.time() - start)

next:
    - create pytable for t-sne data.
    - store example .h5 file.
    - introduce threading.
        - synchronize threads, e. g. with locks/rlocks/barriers. threads add to hd5-/pytable
        - alternative: just put results in shared collection. have another thread check that collection periodically and
          dump results to file.