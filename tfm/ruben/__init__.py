from heapq import nlargest
import numpy as np
import pickle


def _point_box_crosses_diagonal(point, c):
    bottom_right_corner = (point.birth + c, point.death + c)
    return bottom_right_corner[0] >= bottom_right_corner[1]


# DEPRECATED FOR DIPHA
class PersistenceDiagramPoint:
    def __init__(self, dim, birth, death):
        self.dim = dim
        self.birth = birth
        self.death = death


class PersistenceDiagram:
    def __init__(self):
        self.points = []

    def add_point(self, persistence_diagram_point):
        self.points.append(persistence_diagram_point)

    def get_thresholded_points(self, persistence_threshold):
        return list(filter(lambda point: point.death - point.birth > persistence_threshold, self.points))

    def get_life(self, dim=None):
        difference = map(lambda point: point.death - point.birth, self.points)
        # Filtering the infinites
        diff_without_inf = filter(lambda diff: not np.isinf(diff), difference)
        return np.mean(np.fromiter(diff_without_inf, dtype=np.double))

    def get_midlife(self):
        mean = map(lambda point: (point.death +
                   point.birth) / 2.0, self.points)
        # Filtering the infinites
        mean_without_inf = filter(lambda diff: not np.isinf(diff), mean)
        return np.mean(np.fromiter(mean_without_inf, dtype=np.double))

    def get_cleaned_persistence_diagram(self, c):
        cleaned_persistence_diagram = PersistenceDiagram()
        significant_points = filter(
            lambda point: _point_box_crosses_diagonal(point, c), self.points)
        for point in significant_points:
            cleaned_persistence_diagram.add_point(point)
        return cleaned_persistence_diagram

    def get_finite_points(self):
        return list(filter(lambda point: (not np.isinf(point.birth)) and (not np.isinf(point.death)), self.points))

    def get_topological_pool(self, n=None):
        finite_points = self.get_finite_points()
        prominences = list(
            map(lambda point: point.death - point.birth, finite_points))
        return nlargest(n, prominences)

    def get_finite_persistence_diagram(self):
        finite_points_pd = self.get_finite_points()
        finite_pd = PersistenceDiagram()
        for point in finite_points_pd:
            finite_pd.add_point(PersistenceDiagramPoint(
                point.dim, point.birth, point.death))
        return finite_pd

    def get_persistence_diagram_of_dim(self, dim):
        points_of_dim = list(
            filter(lambda point: point.dim == dim, self.points))
        pd_of_dim = PersistenceDiagram()
        for point in points_of_dim:
            pd_of_dim.add_point(PersistenceDiagramPoint(
                point.dim, point.birth, point.death))
        return pd_of_dim

    def save(self, filename):
        representation = [(p.dim, p.birth, p.death) for p in self.points]
        with open(filename, "wb") as f:
            pickle.dump(representation, f)

    @classmethod
    def load(cls, filename):
        pd = cls()
        with open(filename, "rb") as f:
            points = pickle.load(f)
        for p in points:
            pd.add_point(PersistenceDiagramPoint(*p))
        return pd

    def __iter__(self):
        return iter(self.points)
