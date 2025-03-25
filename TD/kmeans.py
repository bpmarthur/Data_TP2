import numpy as np
import random

class Point:
    """A point in a dataset.

    Attributes:
        d: int             -- dimension of the ambient space
        coords: np.ndarray -- coordinates of the point
        label: int = 0     -- label of the cluster the point is assigned to
    """
    #To create an nd array we use np.array

    def __init__(self, d: int):
        assert d > 0
        self.coords = np.array([0.0 for _ in range(d)])  #np.zeros works too
        self.label = 0

    def update_coords(self, new_coords: np.ndarray) -> None:
        self.coords = np.copy(new_coords)

    def squared_dist(self, other) -> float:
        return np.sum((self.coords - other.coords) ** 2)
    
        #numpy.sum(a, axis=None) Sum of array elements over a given axis.

    
class Cloud:
    """A cloud of points for the k-means algorithm.

    Data attributes:
    - d: int              -- dimension of the ambient space
    - points: list[Point] -- list of points
    - k: int              -- number of centers
    - centers: np.ndarray -- array of centers
    """

    def __init__(self, d: int, k: int):
        self.d = d
        self.k = k
        self.points = []
        self.centers = np.array([Point(d) for _ in range(self.k)])

    def add_point(self, p: Point, label: int) -> None:
        """Copy p to the cloud, in cluster label."""
        new_point = Point(self.d)
        new_point.update_coords(p.coords)
        #Ici on copie le point p pour ne pas avoir de référence partagée !
        self.points.append(new_point)
        self.points[-1].label = label

    def intracluster_variance(self) -> float:
        variance = 0.0
        for i in range(self.k):
            for p in self.points:
                if p.label == i:
                    variance += p.squared_dist(self.centers[i])
        return variance/self.points.__len__()
    
        '''
        #Essai d'algorithme de manière linéaire
        variances = np.zeros(self.k)
        for p in self.points:
            variances[p.label] += p.squared_dist(self.centers[p.label])
        return np.sum(variances)/self.points.__len__()
        '''
    def set_voronoi_labels(self) -> int:
        nb_changes = 0  #Nombre de changements de labels
        for p in self.points:
            min_dist = p.squared_dist(self.centers[0])
            new_label = 0
            for i in range(1, self.k):
                dist = p.squared_dist(self.centers[i])
                if dist < min_dist:
                    min_dist = dist
                    new_label = i
            if new_label != p.label:
                nb_changes += 1
                p.label = new_label
        return nb_changes
    
    def set_centroid_centers(self) -> None:
        label_count = np.zeros(self.k)
        new_centers = np.array([Point(self.d) for _ in range(self.k)])
        for i in range(self.k):
            new_centers[i].update_coords(np.array([0.0 for _ in range(self.d)]))
        for p in self.points:
            label_count[p.label] += 1
            new_centers[p.label].coords += p.coords
        for i in range(self.k):
            if label_count[i] > 0:
                new_centers[i].coords /= label_count[i]
            else :
                new_centers[i].coords = self.centers[i].coords
        self.centers = new_centers
        return
    
    def init_random_partition(self) -> None:
        for p in self.points:
            p.label = random.randrange(self.k)
        self.set_centroid_centers()
        return

    def lloyd(self) -> None:
        """Lloyd’s algorithm.
        Assumes the clusters have already been initialized somehow.
        """
        while(self.set_voronoi_labels() > 0):
            self.set_centroid_centers()
        return
    
    def init_forgy(self) -> None:
        """Forgy's initialization: distinct centers are sampled
        uniformly at random from the points of the cloud.
        """
        seen = np.array([False for _ in range(len(self.points))])
        for i in range(self.k):
            j = random.randrange(len(self.points))
            while seen[j]:
                j = random.randrange(len(self.points))
            self.centers[i].update_coords(self.points[j].coords)
            seen[j] = True
        return

    
    def init_plusplus(self) -> None:
        centers = [Point(self.d) for _ in range(self.k)]    #Tableau des centres que l'on va choisir
        centers[0] = self.points[random.randrange(len(self.points))]    #On choisit un premier centre aléatoire
        for i in range(1, self.k):
            distances = np.array([min([p.squared_dist(c) for c in centers[:i]]) for p in self.points])  #Il est important de mettre centers[:i] pour ne pas prendre en compte les centres non encore choisis
            probas = distances/np.sum(distances)
            cum_probas = np.cumsum(probas)
            r = random.uniform(0.0, 1.0)
            j = 0
            while r > cum_probas[j]:
                j += 1
            centers[i] = self.points[j]   #On choisit le point j comme centre
        self.centers = centers
        return
        #random.uniform(0.0, 1.0)