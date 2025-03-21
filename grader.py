#! /usr/bin/env python3
import sys
import unittest
import numpy as np

from itertools import permutations

from TD.kmeans import *
from TD.kmeans_application import *


"""
Annotations used for the autograder.

[START-AUTOGRADER-ANNOTATION]
{
  "total" : 9,
  "names" : [
      "kmeans.py::test_point",
      "kmeans.py::test_intracluster_variance",
      "kmeans.py::test_voronoi",
      "kmeans.py::test_centroids",
      "kmeans.py::test_init_random_partition",
      "kmeans.py::test_lloyd",
      "kmeans.py::test_forgy",
      "kmeans.py::test_init_plusplus",
      "kmeans_application.py::kaggle_exercise"
      ],
  "points" : [5, 10, 15, 15, 10, 15, 10, 10, 10]
}
[END-AUTOGRADER-ANNOTATION]
"""


def print_help():
    print(
        "./grader script. Usage: ./grader.py test_number, e.g., ./grader.py 1 for the 1st exercise."
    )
    print("N.B.: ./grader.py 0 runs all tests.")
    print(f"You provided {sys.argv}.")
    exit(1)


class Grader(unittest.TestCase):
    def test_point(self):
        d = 2
        p = Point(d)
        q = Point(d)

        ops = np.array([1, 2])
        q.coords[0] += ops[0]
        q.coords[0] /= ops[1]
        self.assertAlmostEqual(q.coords[0], 0.5, msg="point constructor")
        q.coords[0] = 0

        self.assertAlmostEqual(p.coords[0], 0.0, msg="point constructor")
        self.assertAlmostEqual(p.coords[1], 0.0, msg="point constructor")

        r = np.array([-1.0, 1.0])
        q.update_coords(r)
        self.assertAlmostEqual(q.coords[0], -1.0, msg="update_coords")
        self.assertAlmostEqual(q.coords[1], 1.0, msg="update_coords")
        r[0] = 0.0
        self.assertAlmostEqual(
            q.coords[0], -1.0, msg="update_coords: Argument probably not copied."
        )

        self.assertAlmostEqual(q.squared_dist(q), 0.0, msg="squared_dist")
        self.assertAlmostEqual(p.squared_dist(q), 2.0, msg="squared_dist")
        self.assertAlmostEqual(q.squared_dist(p), 2.0, msg="squared_dist")

    def test_intracluster_variance(self):
        p = Point(1)

        # 1
        sol = 0.0
        c = Cloud(1, 1)
        c.add_point(p, 0)
        self.assertAlmostEqual(
            c.intracluster_variance(), sol, msg="intracluster-variance"
        )

        # 2
        sol = 0.25
        c = Cloud(1, 1)
        p.coords[0] = 0.5
        c.add_point(p, 0)

        self.assertAlmostEqual(
            c.intracluster_variance(), sol, msg="intracluster-variance"
        )

        # 3
        sol = 0.625
        c = Cloud(1, 1)
        p.coords[0] = -1.0
        c.add_point(p, 0)

        p.coords[0] = 0.5
        c.add_point(p, 0)

        p.coords[0] = -0.5
        c.centers[0] = p

        self.assertAlmostEqual(
            c.intracluster_variance(), sol, msg="intracluster-variance"
        )

        # 4
        sol = 6.8125
        c = Cloud(1, 2)
        p.coords[0] = -1.0
        c.add_point(p, 0)

        p.coords[0] = 0.5
        c.add_point(p, 0)

        p.coords[0] = -0.5
        c.centers[0] = p

        q = Point(1)
        q.coords[0] = -2.0
        c.add_point(q, 1)

        q.coords[0] = 2.0
        c.add_point(q, 1)

        q.coords[0] = -3.0
        c.centers[1] = q

        self.assertAlmostEqual(
            c.intracluster_variance(), sol, msg="intracluster-variance"
        )

    def test_voronoi(self):
        # 1
        c = Cloud(1, 2)
        p = Point(1)

        p.coords[0] = -3.0
        c.add_point(p, 1)

        p.coords[0] = -1.0
        c.add_point(p, 1)

        p.coords[0] = 0.0
        c.add_point(p, 1)

        p.coords[0] = -2.0
        c.centers[0] = p

        q = Point(1)
        q.coords[0] = 1.9
        c.centers[1] = q

        nb = c.set_voronoi_labels()

        self.assertEqual(c.points[0].label, 0, msg="set_voronoi_labels -- labels")
        self.assertEqual(c.points[1].label, 0, msg="set_voronoi_labels -- labels")
        self.assertEqual(c.points[2].label, 1, msg="set_voronoi_labels -- labels")
        self.assertEqual(nb, 2, msg="set_voronoi_labels -- return value")

        # 2
        c = Cloud(2, 2)
        p = Point(2)

        p.coords[0] = -3.0
        p.coords[1] = 3.0
        c.add_point(p, 1)

        p.coords[0] = -1.0
        p.coords[1] = 0.0
        c.add_point(p, 1)

        p.coords[0] = 2.0
        p.coords[1] = 1.0
        c.add_point(p, 1)

        p.coords[0] = -5.0
        p.coords[1] = -2.0
        c.centers[0] = p

        q = Point(2)
        q.coords[0] = 0.0
        q.coords[1] = -2.0
        c.centers[1] = q

        nb = c.set_voronoi_labels()

        self.assertEqual(c.points[0].label, 0, msg="set_voronoi_labels -- labels")
        self.assertEqual(c.points[1].label, 1, msg="set_voronoi_labels -- labels")
        self.assertEqual(c.points[2].label, 1, msg="set_voronoi_labels -- labels")
        self.assertEqual(nb, 1, msg="set_voronoi_labels -- return value")

    def test_centroids(self):
        # 1
        c = Cloud(1, 3)

        p = Point(1)
        p.coords[0] = -3.0
        c.add_point(p, 0)
        c.centers[0] = p

        q = Point(1)
        q.coords[0] = -1.0
        c.add_point(q, 1)
        c.centers[1] = q

        r = Point(1)
        r.coords[0] = -2.1
        c.add_point(r, 1)

        r.coords[0] = 7.6
        c.centers[2] = r

        c.set_centroid_centers()

        self.assertAlmostEqual(c.centers[0].coords[0], -3.0, msg="set_centroid_centers")
        self.assertAlmostEqual(
            c.centers[1].coords[0], -1.55, msg="set_centroid_centers"
        )
        self.assertAlmostEqual(c.centers[2].coords[0], 7.6, msg="set_centroid_centers")

        # 2
        c = Cloud(1, 2)

        p = Point(1)
        p.coords[0] = -3.0
        c.add_point(p, 0)
        c.centers[0] = p

        q = Point(1)
        q.coords[0] = -1.0
        c.add_point(q, 1)
        c.centers[1] = q

        r = Point(1)
        r.coords[0] = -2.1
        c.add_point(r, 1)

        c.set_voronoi_labels()
        c.set_centroid_centers()

        self.assertAlmostEqual(
            c.centers[0].coords[0], -2.55, msg="set_centroid_centers"
        )
        self.assertAlmostEqual(c.centers[1].coords[0], -1.0, msg="set_centroid_centers")

        # 3
        c = Cloud(2, 2)

        p = Point(2)
        p.coords[0] = -3.0
        c.add_point(p, 0)
        c.centers[0] = p

        q = Point(2)
        q.coords[0] = -1.0
        c.add_point(q, 1)
        c.centers[1] = q

        r = Point(2)
        r.coords[0] = -2.1
        c.add_point(r, 1)

        c.set_voronoi_labels()
        c.set_centroid_centers()

        self.assertAlmostEqual(
            c.centers[0].coords[0], -2.55, msg="set_centroid_centers"
        )
        self.assertAlmostEqual(c.centers[1].coords[0], -1.0, msg="set_centroid_centers")

    def test_init_random_partition(self):
        K = 10000  # number of random experiments
        delta = 0.0625  # tolerance used in probability
        p = Point(1)
        three_points = Cloud(1, 3)
        prob_three_points = 0.3333

        p.coords[0] = 0.0
        three_points.add_point(p, 0)

        p.coords[0] = 1.0
        three_points.add_point(p, 0)

        p.coords[0] = 2.0
        three_points.add_point(p, 0)

        count = 0
        for _ in range(K):
            three_points.init_random_partition()
            if three_points.points[2].label == 1:
                count += 1

        self.assertAlmostEqual(
            count / K,
            prob_three_points,
            msg="random partition -- wrong frequency of choosing a center",
            delta=delta,
        )

    def test_lloyd(self):
        c = Cloud(2, 2)
        p = Point(2)

        p.coords[0] = 1.0
        p.coords[1] = 1.0
        c.add_point(p, 0)

        p.coords[0] = 2.0
        p.coords[1] = 1.0
        c.add_point(p, 1)

        p.coords[0] = 4.0
        p.coords[1] = 3.0
        c.add_point(p, 1)

        p.coords[0] = 5.0
        p.coords[1] = 4.0
        c.add_point(p, 1)

        p.coords[0] = 1.0
        p.coords[1] = 1.0
        c.centers[0] = p

        q = Point(2)
        q.coords[0] = 2.0
        q.coords[1] = 1.0
        c.centers[1] = q

        c.set_centroid_centers()
        c.lloyd()

        if (c.centers[0].coords[0] - 1.5) <= 0.0001:
            self.assertAlmostEqual(c.centers[0].coords[0], 1.5, msg="lloyd -- centers")
            self.assertAlmostEqual(c.centers[1].coords[0], 4.5, msg="lloyd -- centers")
            self.assertEqual(c.points[0].label, 0, msg="lloyd -- labels")
            self.assertEqual(c.points[1].label, 0, msg="lloyd -- labels")
            self.assertEqual(c.points[2].label, 1, msg="lloyd -- labels")
        else:
            self.assertAlmostEqual(c.centers[0].coords[0], 4.5, msg="lloyd -- centers")
            self.assertAlmostEqual(c.centers[1].coords[0], 1.5, msg="lloyd -- centers")
            self.assertEqual(c.points[0].label, 1, msg="lloyd -- labels")
            self.assertEqual(c.points[1].label, 1, msg="lloyd -- labels")
            self.assertEqual(c.points[2].label, 0, msg="lloyd -- labels")

    def test_forgy(self):
        K = 10000  # number of random experiments
        delta = 0.0625  # tolerance used in probability
        p = Point(1)
        three_points = Cloud(1, 2)
        prob_three_points = 0.3333

        p.coords[0] = 0.0
        three_points.add_point(p, 0)

        p.coords[0] = 1.0
        three_points.add_point(p, 0)

        p.coords[0] = 2.0
        three_points.add_point(p, 0)

        count = 0
        is_unique = True
        for _ in range(K):
            three_points.init_forgy()
            if three_points.centers[0].coords[0] == 1.0:
                count += 1
            if three_points.centers[0].coords[0] == three_points.centers[1].coords[0]:
                is_unique = False

        self.assertAlmostEqual(
            count / K,
            prob_three_points,
            msg="forgy -- wrong frequency of choosing a center",
            delta=delta,
        )
        self.assertEqual(is_unique, True, msg="forgy -- center not unique")

    def test_init_plusplus(self):
        K = 10000  # number of random experiments
        delta = 0.0625  # tolerance used in probability
        p = Point(1)

        # 1
        three_points = Cloud(1, 1)
        prob_three_points = 0.3333

        p.coords[0] = 0.0
        three_points.add_point(p, 0)

        p.coords[0] = 1.0
        three_points.add_point(p, 0)

        p.coords[0] = 2.0
        three_points.add_point(p, 0)

        count = 0
        for _ in range(K):
            three_points.init_plusplus()
            if three_points.centers[0].coords[0] == 1.0:
                count += 1

        self.assertAlmostEqual(
            count / K,
            prob_three_points,
            msg="plusplus -- wrong frequency of choosing a center",
            delta=delta,
        )

        # 2
        two_clusters = Cloud(1, 2)
        prob_two_clusters = 0.125

        p.coords[0] = 0.0
        two_clusters.add_point(p, 0)

        p.coords[0] = 0.0
        two_clusters.add_point(p, 0)

        p.coords[0] = 1.0
        two_clusters.add_point(p, 0)

        p.coords[0] = 2.0
        two_clusters.add_point(p, 0)

        count = 0
        for _ in range(K):
            two_clusters.init_plusplus()
            if two_clusters.centers[1].coords[0] == 1.0:
                count += 1

        self.assertAlmostEqual(
            count / K,
            prob_two_clusters,
            msg="plusplus -- wrong frequency of choosing a center",
            delta=delta,
        )

    def kaggle_exercise(self):
        # Read data
        data_labeled = np.genfromtxt(
            "../samples_full.csv", delimiter=","
        )  # This path needs to look like this on grader!
        labels = (data_labeled[:, -1:].transpose()[0]).astype(int)
        data = data_labeled[:, :-1]
        (nmax, d) = data.shape

        # Your data
        (nb_clusters, my_labels, my_center_coords) = my_parametrization(d, nmax)

        # Create cloud
        cloud = Cloud(d, nb_clusters)
        for i in range(nmax):
            p = Point(d)
            p.update_coords(data[i])
            cloud.add_point(p, my_labels[i])

        # Perform clustering
        for i in range(nb_clusters):
            cloud.centers[i].update_coords(my_center_coords[i])
            cloud.centers[i].label = i
        cloud.lloyd()

        # Test accuracy
        label_map = list(range(nb_clusters))
        best_score = -1
        for curr_map in permutations(label_map):
            correct_labels = 0
            for i in range(nmax):
                if curr_map[cloud.points[i].label] == labels[i]:
                    correct_labels += 1
            if correct_labels > best_score:
                best_score = correct_labels
        print("Correctly predicted: ", best_score)

        # Grade the outcome
        match best_score:
            case c if 0 <= c <= 120:
                print("D")
            case c if 121 <= c <= 500:
                print("C")
            case c if 501 <= c <= 800:
                print("B")
            case c if 801 <= c <= 1199:
                print("A")
            case c if c >= 1200:
                print("A+")


def suite(test_nb):
    suite = unittest.TestSuite()
    test_name = [
        "test_point",
        "test_intracluster_variance",
        "test_voronoi",
        "test_centroids",
        "test_init_random_partition",
        "test_lloyd",
        "test_forgy",
        "test_init_plusplus",
        "kaggle_exercise",
    ]

    if test_nb > 0:
        suite.addTest(Grader(test_name[test_nb - 1]))
    else:
        for name in test_name:
            suite.addTest(Grader(name))

    return suite


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_help()
    try:
        test_nb = int(sys.argv[1])
    except ValueError as e:
        print(
            f"You probably didn't pass an int to ./grader.py: passed {sys.argv[1]}; error {e}"
        )
        exit(1)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite(test_nb))
