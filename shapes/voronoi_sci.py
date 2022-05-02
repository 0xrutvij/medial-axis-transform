# from scipy.spatial import ConvexHull, convex_hull_plot_2d, Voronoi, voronoi_plot_2d
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, deque

plt.rcParams["figure.figsize"] = (8,8)

# https://stackoverflow.com/a/65105672
def compute_triangle_circumcenters(xy_pts, tri_arr):
    """
    Compute the centers of the circumscribing circle of each triangle in a triangulation.
    :param np.array xy_pts : points array of shape (n, 2)
    :param np.array tri_arr : triangles array of shape (m, 3), each row is a triple of indices in the xy_pts array

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

    v1 = (tri_pts[:, 1, 0] ** 2 + tri_pts[:, 1, 1] ** 2) - (tri_pts[:, 0, 0] ** 2 + tri_pts[:, 0, 1] ** 2)
    v2 = (tri_pts[:, 2, 0] ** 2 + tri_pts[:, 2, 1] ** 2) - (tri_pts[:, 0, 0] ** 2 + tri_pts[:, 0, 1] ** 2)

    # solve 2x2 system (see https://en.wikipedia.org/wiki/Invertible_matrix#Inversion_of_2_%C3%97_2_matrices)
    det = (a * d - b * c)
    detx = (v1 * d - v2 * b)
    dety = (a * v2 - c * v1)

    x = detx / det
    y = dety / det

    return (np.vstack((x, y))).T

def in_circle(d1, d2, v1, v2):
    xd1, yd1 = d1[0], d1[1]
    xd2, yd2 = d2[0], d2[1]
    xv1, yv1 = v1[0], v1[1]
    xv2, yv2 = v2[0], v2[1]

    matrix = [
        [xd1, yd1, xd1 ** 2 + yd1 ** 2, 1],
        [xd2, yd2, xd2 ** 2 + yd2 ** 2, 1],
        [xv1, yv1, xv1 ** 2 + yv1 **2, 1],
        [xv2, yv2, xv2 ** 2 + yv2 ** 2, 1]
    ]

    return np.linalg.det(matrix)

input_file = "/Users/rutvijshah/CodeBase/ComputationalGeometry/PyCg/Shapes/horse-1.txt"
points = []
xy_points = [[], []]

with open(input_file, "r") as infile:
    lines = infile.readlines()
    for line in lines:
        if line.startswith("# points"):
            continue
        elif line.startswith("# triangles"):
            break
        pts_str = line.split(" ")
        x, y = map(lambda x: float(x) * 10, pts_str)
        xy_points[0].append(x)
        xy_points[1].append(y)
        points.append([x, y])

consec_edges = []
for p1, p2 in zip(points[:-1], points[1:]):
    consec_edges.append([p1, p2])

points = np.array(points)
num_points = len(points)
# vor = Voronoi(points)
tri = Delaunay(points)

process_queue = deque([])
face_classification = {}


def is_consec_delaunay_edge(i, j):
    return abs(i - j) == 1 or abs(i - j) == num_points - 2

for i in range(len(tri.simplices)):
    current_simplex = tri.simplices[i,:]
    # classifying triangle, this case only those having a neighbor
    # face as the infinite Delaunay face
    neighbors = tri.neighbors[i]
    neighbors_simplices = tri.simplices[neighbors]

    neighbors = tri.neighbors[i]
    for j, neigh in enumerate(neighbors):
        if neigh == -1:
            shared_edge = np.setdiff1d(current_simplex, [current_simplex[j]])
            # print(current_simplex)
            # print(neighbors)
            # print(shared_edge)
            if is_consec_delaunay_edge(*tuple(shared_edge)):
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
    current_simplex = tri.simplices[f,:]
    neighbors = tri.neighbors[f]
    neighbors_simplices = tri.simplices[neighbors]
    self_class = face_classification[f]

    for neighbor, its_simplex in zip(neighbors, neighbors_simplices):
        if neighbor != -1:
            common_edge = np.intersect1d(current_simplex, its_simplex)
            if is_consec_delaunay_edge(*tuple(common_edge)):
                face_classification[neighbor] = "inside" if self_class == "outside" else "outside"
            else:
                face_classification[neighbor] = self_class
            process_queue.append(neighbor)


cc = compute_triangle_circumcenters(tri.points, tri.simplices)

skeleton_edges = []
point_radii_pair = {}

for i in range(len(tri.simplices)):
    current_simplex = tri.simplices[i,:]
    v1 = cc[i]
    r1 = np.linalg.norm(v1 - tri.points[current_simplex[0]])
    self_class = face_classification[i]

    neighbors = tri.neighbors[i]
    for neigh in neighbors:
        if neigh != -1:
            neighbor_simplex = tri.simplices[neigh,:]
            # common_edge = np.intersect1d(current_simplex, neighbor_simplex)
            # d1, d2 = tri.points[common_edge[0]], tri.points[common_edge[1]]

            v2 = cc[neigh]
            r2 = np.linalg.norm(v2 - tri.points[neighbor_simplex[0]])
            neigh_class = face_classification[neigh]

            # if in_circle(d1, d2, v1, v2) >= 0:
            if self_class == "inside" and neigh_class == "inside":
                point_radii_pair[tuple(v1)] = r1
                point_radii_pair[tuple(v2)] = r2
                skeleton_edges.append([v1, v2])


plt.plot(points[:,0], points[:,1], 'o')

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

skeleton_edges = np.array(skeleton_edges)
graph_nodes_all = skeleton_edges.reshape((skeleton_edges.size // 2, 2))
graph_nodes = np.unique(graph_nodes_all, axis=0)

plt.gca().set_xlim([-1, 11])
plt.gca().set_ylim([-1, 11])
plt.show()


# print(point_radii_pair.items())
fig, ax = plt.subplots()
for p, r in point_radii_pair.items():
    circ = plt.Circle(p, r, color='b', fill=True)
    ax.add_patch(circ)

plt.gca().set_xlim([-1, 11])
plt.gca().set_ylim([-1, 11])
plt.show()
# end point : one adjacent point
# junction point : three or more adjacent points
# connection point : none of the above.


for node in graph_nodes:
    pass
