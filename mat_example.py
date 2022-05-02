import argparse
import matplotlib.pyplot as plt
import numpy as np

from modified_skeleton_algorithm import MedialAxisTransformer
from scipy.spatial import Delaunay
from utils import extract_points_from_file

# skeleton_edges = np.array(skeleton_edges)
# graph_nodes_all = skeleton_edges.reshape((skeleton_edges.size // 2, 2))
# graph_nodes = np.unique(graph_nodes_all, axis=0)


# end point : one adjacent point
# junction point : three or more adjacent points
# connection point : none of the above.

if __name__ == "__main__":

    plt.rcParams["figure.figsize"] = (8, 8)

    parser = argparse.ArgumentParser(description="A script to extract the MAT from a shape's point set.")
    parser.add_argument(
        "-f", "--file",
        type=str,
        help="file containing points as x-y pairs (format same as that for files in shapes folder)",
        default="shapes/horse-1.txt"
    )

    args = parser.parse_args()
    input_file = args.file

    points = extract_points_from_file(input_file)

    points = np.array(points)
    plt.plot(points[:,0], points[:,1], 'o')

    num_points = len(points)
    tri = Delaunay(points)

    ma_skeleton = MedialAxisTransformer.from_delaunay_triangulation(tri)

    skeleton_edges = ma_skeleton.medial_axis_edges
    point_radii_pair = ma_skeleton.medial_points_radius_pairs

    visited = set()
    for edge in skeleton_edges:
        start, end = edge
        xs, ys = [start[0], end[0]], [start[1], end[1]]
        edge_key = (tuple(xs), tuple(ys))
        if edge_key not in visited:
            visited.add(edge_key)
        else:
            continue
        plt.plot(xs, ys, "b")

    plt.gca().set_xlim([-1, 11])
    plt.gca().set_ylim([-1, 11])
    plt.show()

    fig, ax = plt.subplots()
    for p, r in point_radii_pair.items():
        circ = plt.Circle(p, r, color='b', fill=True)
        ax.add_patch(circ)

    plt.gca().set_xlim([-1, 11])
    plt.gca().set_ylim([-1, 11])
    plt.show()
