import random
import numpy as np

def my_parametrization(d: int, nb_samples: int) -> (int, np.ndarray, np.ndarray):
    """Set empirically determined initial parameters for $k$-means."""
    nb_clusters = 4
    my_labels = np.array([random.randint(0, nb_clusters-1) for _ in range(nb_samples)])
    my_center_coords = [np.array([random.uniform(0, 1) for _ in range(d)]) for _ in range(nb_clusters)]

    '''
    Le nombre nb_clusters = 4 est le nombre de clusters qui minimise la variance intra-cluster pour les données d'entraînement
    (samples_excerpt.csv). Les labels my_labels sont attribués aléatoirement à chaque point, et cela ne pose pas de problème car dans
    kmeans_performance_check, on appelle notre fonction kmeans.Cloud lloyd() qui actualise les labels des points au centroïde le plus proche.
    '''

    return (nb_clusters, my_labels, my_center_coords)
