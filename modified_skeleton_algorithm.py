from scipy.spatial import Delaunay
from collections import defaultdict, deque
from utils import TwoWayDict, compute_triangle_circumcenters, is_consec_delaunay_edge, in_circle
from typing import List, Dict, Tuple

import numpy as np


class MedialAxisTransformer:

    def __init__(self):
        self.medial_axis_edges = []
        self.medial_points_radius_pairs = {}
        self.medial_axis_point_idx_map = TwoWayDict(Tuple, int)
        self._endpoints: List = None
        self._adjacency_map: Dict[int, Dict] = None

    def _set_medial_axis_point_mapping(self):
        skeleton_edges = np.array(self.medial_axis_edges)
        dup_point_list = list(skeleton_edges.reshape(
            (skeleton_edges.size // 2, 2)))

        for i, point in enumerate(set(map(tuple, dup_point_list))):
            self.medial_axis_point_idx_map[point] = i

    def get_radii_vector_along_path(self, path_pt_indices: List[int]) -> np.ndarray:
        radii = [self.medial_points_radius_pairs[self.medial_axis_point_idx_map[i]]
                 for i in path_pt_indices]
        r1 = np.array(radii)
        sr1 = sum(r1)
        n = len(radii)
        normalization_div = (sr1 / n)
        return r1 / normalization_div

    @property
    def skeleton_end_point_indices(self) -> List:
        if self._endpoints is None:
            self._create_skeleton_graph()
        return self._endpoints.copy()

    @property
    def skeleton_graph(self) -> Dict[int, Dict]:
        if self._adjacency_map is None:
            self._create_skeleton_graph()
        return self._adjacency_map

    def _create_skeleton_graph(self) -> None:

        adjacency_map = defaultdict(dict)

        for edge in self.medial_axis_edges:
            pt1, pt2 = map(tuple, edge)
            pid1, pid2 = map(
                lambda x: self.medial_axis_point_idx_map[x], (pt1, pt2))

            dist = np.linalg.norm(np.array(pt1) - np.array(pt2))

            adjacency_map[pid1][pid2] = dist
            adjacency_map[pid2][pid1] = dist

        self._endpoints = [point for point,
                           adj_map in adjacency_map.items() if len(adj_map) == 1]

        self._adjacency_map = adjacency_map

    @staticmethod
    def _classify_delaunay_faces(delaunay_triangulation: Delaunay) -> Dict:
        process_queue = deque([])
        face_classification = {}
        num_points = len(delaunay_triangulation.points)

        for i in range(len(delaunay_triangulation.simplices)):
            current_simplex = delaunay_triangulation.simplices[i, :]
            # classifying triangle, this case only those having a neighbor
            # face as the infinite Delaunay face
            neighbors = delaunay_triangulation.neighbors[i]
            neighbors_simplices = delaunay_triangulation.simplices[neighbors]

            neighbors = delaunay_triangulation.neighbors[i]
            for j, neigh in enumerate(neighbors):
                if neigh == -1:
                    shared_edge = np.setdiff1d(
                        current_simplex, [current_simplex[j]])
                    if is_consec_delaunay_edge(*tuple(shared_edge), num_points):
                        face_classification[i] = "inside"
                    else:
                        face_classification[i] = "outside"
                    process_queue.append(i)
                    break

        processed = set()
        while process_queue:
            f = process_queue.popleft()
            if f not in processed:
                processed.add(f)
            else:
                continue
            current_simplex = delaunay_triangulation.simplices[f, :]
            neighbors = delaunay_triangulation.neighbors[f]
            neighbors_simplices = delaunay_triangulation.simplices[neighbors]
            self_class = face_classification[f]

            for neighbor, its_simplex in zip(neighbors, neighbors_simplices):
                if neighbor != -1:
                    common_edge = np.intersect1d(current_simplex, its_simplex)
                    if is_consec_delaunay_edge(*tuple(common_edge), num_points):
                        nclass = "inside" if self_class == "outside" else "outside"
                        face_classification[neighbor] = nclass
                    else:
                        face_classification[neighbor] = self_class
                    process_queue.append(neighbor)

        return face_classification

    @staticmethod
    def _extract_mat_from_triangulation(
        algorithm: str,
        tri: Delaunay,
        cc: np.ndarray,
        face_classification: Dict = None
    ) -> Tuple[List, Dict]:

        skeleton_edges = []
        point_radii_pair = {}

        if algorithm == "modified_one_step":
            condition = face_classification is not None
            assert condition, "Modified One Step algorithm needs classified faces."

        for i in range(len(tri.simplices)):
            current_simplex = tri.simplices[i, :]
            v1 = cc[i]
            r1 = np.linalg.norm(v1 - tri.points[current_simplex[0]])
            self_class = None
            if algorithm == "modified_one_step":
                self_class = face_classification[i]

            neighbors = [x for x in tri.neighbors[i] if x != -1]
            for neigh in neighbors:
                neighbor_simplex = tri.simplices[neigh, :]

                v2 = cc[neigh]
                r2 = np.linalg.norm(v2 - tri.points[neighbor_simplex[0]])

                if algorithm == "modified_one_step":
                    neigh_class = face_classification[neigh]
                    predicate_check = self_class == "inside" and neigh_class == "inside"
                else:
                    common_edge = np.intersect1d(
                        current_simplex, neighbor_simplex)
                    d1, d2 = tri.points[common_edge[0]
                                        ], tri.points[common_edge[1]]
                    predicate_check = in_circle(d1, d2, v1, v2) >= 0

                if predicate_check:
                    point_radii_pair[tuple(v1)] = r1
                    point_radii_pair[tuple(v2)] = r2
                    skeleton_edges.append([v1, v2])

        return skeleton_edges, point_radii_pair

    @classmethod
    def from_delaunay_triangulation(cls,
                                    delaunay_triangulation: Delaunay,
                                    algorithm: str = "modified_one_step"):

        mat_obj = cls()

        if algorithm == "modified_one_step":
            face_classification = cls._classify_delaunay_faces(
                delaunay_triangulation)
        else:
            face_classification = None

        circumcenters = compute_triangle_circumcenters(
            delaunay_triangulation.points,
            delaunay_triangulation.simplices
        )

        mae, mprp = cls._extract_mat_from_triangulation(
            algorithm, delaunay_triangulation,
            circumcenters, face_classification
        )

        mat_obj.medial_axis_edges = mae
        mat_obj.medial_points_radius_pairs = mprp
        mat_obj._set_medial_axis_point_mapping()

        return mat_obj

    @classmethod
    def from_boundary_points(cls, points: List):
        points = np.array(points)
        tri = Delaunay(points)
        return cls.from_delaunay_triangulation(tri)
