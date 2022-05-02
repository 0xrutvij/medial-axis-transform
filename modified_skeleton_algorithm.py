from scipy.spatial import Delaunay
from collections import defaultdict, deque
from utils import compute_triangle_circumcenters, is_consec_delaunay_edge, in_circle
from typing import List, Dict

import numpy as np

class MedialAxisTransformer:

    def __init__(self):
        self.medial_axis_edges = []
        self.medial_points_radius_pairs = {}

    @staticmethod
    def _classify_delaunay_faces(delaunay_triangulation: Delaunay) -> Dict:
        process_queue = deque([])
        face_classification = {}
        num_points = len(delaunay_triangulation.points)

        for i in range(len(delaunay_triangulation.simplices)):
            current_simplex = delaunay_triangulation.simplices[i,:]
            # classifying triangle, this case only those having a neighbor
            # face as the infinite Delaunay face
            neighbors = delaunay_triangulation.neighbors[i]
            neighbors_simplices = delaunay_triangulation.simplices[neighbors]

            neighbors = delaunay_triangulation.neighbors[i]
            for j, neigh in enumerate(neighbors):
                if neigh == -1:
                    shared_edge = np.setdiff1d(current_simplex, [current_simplex[j]])
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
            current_simplex = delaunay_triangulation.simplices[f,:]
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
        cc: Dict,
        face_classification: Dict = None
    ) -> (List, Dict):

        skeleton_edges = []
        point_radii_pair = {}

        if algorithm == "modified_one_step":
            condition = face_classification is not None
            assert condition, "Modified One Step algorithm needs classified faces."

        for i in range(len(tri.simplices)):
            current_simplex = tri.simplices[i,:]
            v1 = cc[i]
            r1 = np.linalg.norm(v1 - tri.points[current_simplex[0]])
            if algorithm == "modified_one_step":
                self_class = face_classification[i]

            neighbors = [x for x in tri.neighbors[i] if x != -1]
            for neigh in neighbors:
                neighbor_simplex = tri.simplices[neigh,:]

                v2 = cc[neigh]
                r2 = np.linalg.norm(v2 - tri.points[neighbor_simplex[0]])

                if algorithm == "modified_one_step":
                    neigh_class = face_classification[neigh]
                    predicate_check = self_class == "inside" and neigh_class == "inside"
                else:
                    common_edge = np.intersect1d(current_simplex, neighbor_simplex)
                    d1, d2 = tri.points[common_edge[0]], tri.points[common_edge[1]]
                    predicate_check = in_circle(d1, d2, v1, v2) >= 0

                if predicate_check:
                    point_radii_pair[tuple(v1)] = r1
                    point_radii_pair[tuple(v2)] = r2
                    skeleton_edges.append([v1, v2])

        return skeleton_edges, point_radii_pair


    @classmethod
    def from_delaunay_triangulation(cls, delaunay_triangulation: Delaunay, algorithm: str = "modified_one_step"):

        mat_obj = cls()

        if algorithm == "modified_one_step":
            face_classification = cls._classify_delaunay_faces(delaunay_triangulation)
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

        return mat_obj
