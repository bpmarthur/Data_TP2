import numpy as np


def my_parametrization(d: int, nb_samples: int) -> (int, np.ndarray, np.ndarray):
    """Set empirically determined initial parameters for $k$-means."""
    nb_clusters = 0
    my_labels = np.zeros(nb_samples, dtype=int)
    my_center_coords = [np.zeros(d) for _ in range(nb_clusters)]
    pass

    return (nb_clusters, my_labels, my_center_coords)
