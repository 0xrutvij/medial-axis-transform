import heapq
from itertools import product
from typing import Dict, Hashable, List, Tuple
import numpy as np


# https://stackoverflow.com/a/65105672

def compute_triangle_circumcenters(xy_pts: np.ndarray, tri_arr: np.ndarray) -> np.ndarray:
    """
    Compute the centers of the circumscribing circle of each triangle in a triangulation.
    :param np.array xy_pts : points array of shape (n, 2)
    :param np.array tri_arr : triangles array of shape (m, 3), each row is a triple of indices in
        the xy_pts array

    :return: circumcenter points array of shape (m, 2)
    """
    tri_pts = xy_pts[tri_arr]  # (m, 3, 2) - triangles as points (not indices)

    # finding the circumcenter (x, y) of a triangle defined by three points:
    # (x-x0)**2 + (y-y0)**2 = (x-x1)**2 + (y-y1)**2
    # (x-x0)**2 + (y-y0)**2 = (x-x2)**2 + (y-y2)**2
    #
    # becomes two linear equations (squares are canceled):
    # 2(x1-x0)*x + 2(y1-y0)*y = (x1**2 + y1**2) - (x0**2 + y0**2)
    # 2(x2-x0)*x + 2(y2-y0)*y = (x2**2 + y2**2) - (x0**2 + y0**2)
    a = 2 * (tri_pts[:, 1, 0] - tri_pts[:, 0, 0])
    b = 2 * (tri_pts[:, 1, 1] - tri_pts[:, 0, 1])
    c = 2 * (tri_pts[:, 2, 0] - tri_pts[:, 0, 0])
    d = 2 * (tri_pts[:, 2, 1] - tri_pts[:, 0, 1])

    v1 = ((tri_pts[:, 1, 0] ** 2 + tri_pts[:, 1, 1] ** 2) -
          (tri_pts[:, 0, 0] ** 2 + tri_pts[:, 0, 1] ** 2))
    v2 = ((tri_pts[:, 2, 0] ** 2 + tri_pts[:, 2, 1] ** 2) -
          (tri_pts[:, 0, 0] ** 2 + tri_pts[:, 0, 1] ** 2))

    det = (a * d - b * c)
    detx = (v1 * d - v2 * b)
    dety = (a * v2 - c * v1)

    x = detx / det
    y = dety / det

    return (np.vstack((x, y))).T


def are_ccw(p1, p2, p3):
    y21 = p2[1] - p1[1]
    x21 = p2[0] - p1[0]
    y32 = p3[1] - p2[1]
    x32 = p3[0] - p2[0]

    diff = (y21 * x32) - (x21 * y32)

    return diff < 0


def in_circle(d1: np.ndarray, d2: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
    if not are_ccw(d1, d2, v1):
        v1, v2 = v2, v1

    xd1, yd1 = d1[0], d1[1]
    xd2, yd2 = d2[0], d2[1]
    xv1, yv1 = v1[0], v1[1]
    xv2, yv2 = v2[0], v2[1]

    matrix = [
        [xd1, yd1, xd1 ** 2 + yd1 ** 2, 1],
        [xd2, yd2, xd2 ** 2 + yd2 ** 2, 1],
        [xv1, yv1, xv1 ** 2 + yv1 ** 2, 1],
        [xv2, yv2, xv2 ** 2 + yv2 ** 2, 1]
    ]

    matrix = np.array(matrix)

    return float(np.linalg.det(matrix))


def extract_points_from_file(file_name: str) -> List:
    points = []
    with open(file_name, "r") as infile:
        lines = infile.readlines()
        for line in lines:
            if line.startswith("# points"):
                continue
            elif line.startswith("# triangles"):
                break
            pts_str = line.split(" ")
            x, y = map(lambda x: float(x) * 10, pts_str)
            points.append([x, y])

    return points


def is_consec_delaunay_edge(i, j, list_len):
    return abs(i - j) == 1 or abs(i - j) == list_len - 2


class TwoWayDict:

    def __init__(self, T1: Hashable, T2: Hashable) -> None:
        assert issubclass(T1, Hashable) and issubclass(
            T2, Hashable), "Keys and values of a TwoWayDict must be Hashable."
        self.T1 = T1
        self.T2 = T2
        self._mapping: Dict[T1, T2] = {}
        self._inverse_mapping: Dict[T2, T1] = {}

    def __getitem__(self, __o):
        if isinstance(__o, self.T1):
            return self._mapping[__o]
        elif isinstance(__o, self.T2):
            return self._inverse_mapping[__o]
        else:
            raise TypeError(
                f"TwoWayDict keys must of type {self.T1} or {self.T2}.")

    def __setitem__(self, __o, __v) -> None:
        if isinstance(__o, self.T1) and isinstance(__v, self.T2):
            self._mapping[__o] = __v
            self._inverse_mapping[__v] = __o
        elif isinstance(__o, self.T2) and isinstance(__v, self.T1):
            self._inverse_mapping[__o] = __v
            self._mapping[__v] = __o
        else:
            raise TypeError(
                f"TwoWayDict keys and values must of type {self.T1} or {self.T2}.")

    def __delitem__(self, __o) -> None:
        if isinstance(__o, self.T1):
            __v = self._mapping[__o]
            del self._mapping[__o]
            del self._inverse_mapping[__v]
        elif isinstance(__o, self.T2):
            __v = self._inverse_mapping[__o]
            del self._inverse_mapping[__o]
            del self._mapping[__v]
        else:
            raise TypeError(
                f"TwoWayDict keys must of type {self.T1} or {self.T2}.")

    def __len__(self):
        return len(self._mapping)

    def __contains__(self, __o) -> bool:
        if isinstance(__o, self.T1):
            return __o in self._mapping
        elif isinstance(__o, self.T2):
            return __o in self._inverse_mapping
        else:
            return False


def sort_counterclockwise(points: np.ndarray, centre: np.ndarray = None):
    if centre is None:
        centre = np.mean(points, axis=0)

    diff = points - centre
    angles = np.arctan2(diff[:, 1], diff[:, 0])
    counterclockwise_points = points[np.argsort(angles)]
    return counterclockwise_points


def dijkstras_shortest_paths(adjacency_map: Dict[int, Dict], start_point: int):
    distances = {point: (float("inf"), []) for point in adjacency_map.keys()}
    distances[start_point] = [0, [start_point]]

    priority_queue = [(0, start_point, [start_point])]
    heapq.heapify(priority_queue)

    while priority_queue:
        current_dist, current_pt, current_path = heapq.heappop(priority_queue)

        # if we have already reached a point via a shorter path, skip it
        if current_dist > distances[current_pt][0]:
            continue

        neighbors = adjacency_map[current_pt]

        for neighbor, ndist in neighbors.items():
            dist = current_dist + ndist
            path = current_path + [neighbor]

            # if this route is shorter than any previous routes, add it
            if dist < distances[neighbor][0]:
                distances[neighbor] = (dist, path)
                heapq.heappush(priority_queue, (dist, neighbor, path))

    return distances


def get_shape_paths(endpoints: List, graph: Dict[int, Dict]) -> Dict[Tuple, Tuple[int, List]]:
    all_pairs_end_points_distances = {key: float(
        "inf") if key[0] != key[1] else (0, []) for key in product(endpoints, endpoints)}
    ep_set = set(endpoints)

    while endpoints and any(
        [dist == float("inf")
         for dist in all_pairs_end_points_distances.values()]
    ):
        ep = endpoints.pop()
        shortest_paths = dijkstras_shortest_paths(graph, ep)

        for pt, dist_path in [item for item in shortest_paths.items() if item[0] in ep_set]:
            dist, path = dist_path
            all_pairs_end_points_distances[(ep, pt)] = (dist, path)
            all_pairs_end_points_distances[(pt, ep)] = (
                dist, list(reversed(path)))

    return all_pairs_end_points_distances
