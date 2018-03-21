import numpy
import scipy
import sklearn
from .DistancePreservationObjective import DistancePreservationObjective
import networkx


class Stress(DistancePreservationObjective):
    """
    Calculates stress criterions (Kruskal's stress, Sammon's stress, S stress, quadratic loss).
    """
    def __init__(
            self,
            low_dimensional_data: numpy.ndarray,
            high_dimensional_data: numpy.ndarray,
            distance_metric: str = 'euclidean',
            use_geodesic_distances: bool = False
    ):
        """
        Initiates new pool for stress-related objectives.
        :param low_dimensional_data:
        :param high_dimensional_data:
        :param distance_metric:
        :param use_geodesic_distances:
        """
        super().__init__(
            low_dimensional_data=low_dimensional_data,
            high_dimensional_data=high_dimensional_data,
            distance_metric=distance_metric,
            use_geodesic_distances=use_geodesic_distances
        )

    def compute(self):
        """
        This method allows to compute multiple stress functions:
            * Kruskal stress https://www.researchgate.net/publication/24061688_Nonmetric_multidimensional_scaling_A_numerical_method
            * S stress http://gifi.stat.ucla.edu/janspubs/1977/articles/takane_young_deleeuw_A_77.pdf
            * Sammon stress http://ieeexplore.ieee.org/document/1671271/?reload=true
            * Quadratic Loss
        """
        
        # We retrieve dimensions of the data
        n, m = self._low_dimensional_data.shape

        #  We compute distance matrices in both spaces
        if self._use_geodesic_distances:
            k = 2
            is_connex = False
            while is_connex is False:
                knn = sklearn.neighbors.NearestNeighbors(n_neighbors=k)
                knn.fit(self._low_dimensional_data)
                M = knn.kneighbors_graph(self._low_dimensional_data, mode='distance')
                graph = networkx.from_scipy_sparse_matrix(M)
                is_connex = networkx.is_connected(graph)
                k += 1
            s_uni_distances = networkx.all_pairs_dijkstra_path_length(graph, cutoff=None, weight='weight')
            s_all_distances = numpy.array([numpy.array(a.items())[:, 1] for a in numpy.array(s_uni_distances.items())[:, 1]])
            s_all_distances = (s_all_distances + s_all_distances.T) / 2
            s_uni_distances = scipy.spatial.distance.squareform(s_all_distances)
            s_all_distances = s_all_distances.ravel()
        else:
            s_uni_distances = scipy.spatial.distance.pdist(self._low_dimensional_data)
            s_all_distances = scipy.spatial.squareform(s_uni_distances).ravel()
        l_uni_distances = scipy.spatial.distance.pdist(self._target_data)
        l_all_distances = scipy.spatial.distance.squareform(l_uni_distances).ravel()

        # We set up the measure dict
        measures = dict()

        # 1. Quadratic Loss
        measures['quadratic_loss'] = numpy.square(s_uni_distances - l_uni_distances).sum()

        # 2. Sammon stress
        measures['sammon_stress'] = (1 / s_uni_distances.sum()) * (
            numpy.square(s_uni_distances - l_uni_distances) / s_uni_distances
        ).sum()

        # 3. S stress
        measures['s_stress'] = numpy.sqrt((1 / n) * (
            numpy.square(
                (numpy.square(s_uni_distances) - numpy.square(l_uni_distances)).sum()
            ) / numpy.power(s_uni_distances, 4)
        )).sum()

        # 4. Kruskal stress
        # We reorder the distances under the order of distances in latent space
        s_all_distances = s_all_distances[l_all_distances.argsort()]
        l_all_distances = l_all_distances[l_all_distances.argsort()]
        # We perform the isotonic regression
        iso = sklearn.isotonic.IsotonicRegression()
        s_iso_distances = iso.fit_transform(s_all_distances, l_all_distances)
        # We compute the kruskal stress
        measures['kruskal_stress'] = numpy.sqrt(
            numpy.square(s_iso_distances - l_all_distances).sum() / numpy.square(l_all_distances).sum())

        return measures
