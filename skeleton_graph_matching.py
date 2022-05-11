import json
from timeit import default_timer as timer
from scipy.optimize import linear_sum_assignment
from modified_skeleton_algorithm import MedialAxisTransformer
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from osb import optimal_subsequence_bijection
from utils import (
    extract_points_from_file,
    sort_counterclockwise,
    get_shape_paths,
)

N_SAMPLES = 20


def path_distance(puv_r1: np.ndarray, puv_r2: np.ndarray, l1: float, l2: float, weight_factor=30):
    """Distance between two paths"""
    r_diff_sq = np.square(puv_r1 - puv_r2)
    r_sum = puv_r1 + puv_r2

    ratio = r_diff_sq / (r_sum + 0.00000001)
    addend = np.sum(ratio)
    augend = weight_factor * ((l1 - l2) ** 2 / (l1 + l2))
    return addend + augend


def path_distance_matrix(
    vi: int,
    g: MedialAxisTransformer,
    vj_prime: int,
    g_prime: MedialAxisTransformer
) -> np.ndarray:

    k = len(g.skeleton_end_points)
    n = len(g_prime.skeleton_end_points)
    if k > n:
        g, g_prime = g_prime, g
        vi, vj_prime = vj_prime, vi
        k, n = n, k

    ks = [g.all_pair_eps_shortest_paths[(vi, x)]
          for x in g.skeleton_end_point_indices if x != vi]
    ns = [g_prime.all_pair_eps_shortest_paths[(
        vj_prime, x)] for x in g_prime.skeleton_end_point_indices if x != vj_prime]

    pd_mat = np.zeros((k - 1, n - 1))
    for i, lpi in enumerate(ks):
        l_k, path_k = lpi
        for j, lpj in enumerate(ns):
            l_n, path_n = lpj
            rn1 = g.sampled_radii_vector_along_path(path_k, l_k, N_SAMPLES)
            rn2 = g_prime.sampled_radii_vector_along_path(
                path_n, l_n, N_SAMPLES)
            pd_mat[i, j] = path_distance(rn1, rn2, l_k, l_n)

    return pd_mat


def main1():
    input_file1 = "shapes/horse-1.txt"
    input_file2 = "shapes/horse-19.txt"

    points1 = extract_points_from_file(input_file1)
    points2 = extract_points_from_file(input_file2)

    ma_skeleton1 = MedialAxisTransformer.from_boundary_points(points1)
    ma_skeleton2 = MedialAxisTransformer.from_boundary_points(points2)

    adjacency_map1 = ma_skeleton1.skeleton_graph
    end_points1 = ma_skeleton1.skeleton_end_point_indices

    adjacency_map2 = ma_skeleton2.skeleton_graph
    end_points2 = ma_skeleton2.skeleton_end_point_indices

    eps1 = np.array([ma_skeleton1.medial_axis_point_idx_map[i]
                    for i in end_points1])

    eps1 = sort_counterclockwise(eps1)

    plt.scatter(eps1[:, 0], eps1[:, 1])

    for i, label in enumerate(end_points1):
        plt.annotate(str(label), tuple(eps1[i]))

    plt.show()

    # find shortest path between an endpoint and all the points in the graph
    all_ep_pair_sps1 = get_shape_paths(end_points1.copy(), adjacency_map1)
    all_ep_pair_sps2 = get_shape_paths(end_points2.copy(), adjacency_map2)

    # seen_lengths = defaultdict(list)
    # for ep in product(end_points1, end_points1):
    #     _, path = all_ep_pair_sps1[ep]
    #     if len(path) > 1:
    #         seen_lengths[len(path)].append(ep)

    # for ep in product(end_points2, end_points2):
    #     _, path = all_ep_pair_sps2[ep]
    #     if len(path) in seen_lengths:
    #         print(f"This {ep}: {len(path)}")
    #         print(f"Possible matches {seen_lengths[len(path)]}")

    l1, path1 = all_ep_pair_sps1[(13, 40)]
    l2, path2 = all_ep_pair_sps2[(12, 43)]

    r1 = ma_skeleton1.get_radii_vector_along_path(path1)
    r2 = ma_skeleton2.get_radii_vector_along_path(path2)

    print(path_distance(r1, r2, l1, l2))


def compare_skeletons(mas1: MedialAxisTransformer, mas2: MedialAxisTransformer, comp_name: str):
    end_points1, end_points2 = mas1.skeleton_end_points, mas2.skeleton_end_points
    k = len(end_points1)
    n = len(end_points2)

    if k > n:
        end_points1, end_points2 = end_points2, end_points1
        k, n = n, k

    corr = np.zeros((n, n))
    for i, ep1 in enumerate(mas1.skeleton_end_point_indices):
        for j, ep2 in enumerate(mas2.skeleton_end_point_indices):
            pd_mat = path_distance_matrix(ep1, mas1, ep2, mas2)
            corr[i, j] = optimal_subsequence_bijection(pd_mat)

    const = np.mean(corr)
    if n > k:
        corr[k:n, :] = const

    _, gp = linear_sum_assignment(np.array(corr))
    total_cost = 0
    for i, j in enumerate(gp):
        if i < len(mas1.skeleton_end_point_indices):
            total_cost += corr[i, j]

    return comp_name, total_cost


def main2(f1, f2):
    start = timer()
    g_pts, gprime_pts = map(extract_points_from_file, [f1, f2])
    mas1, mas2 = map(MedialAxisTransformer.from_boundary_points, [
                     g_pts, gprime_pts])
    end_points1, end_points2 = mas1.skeleton_end_points, mas2.skeleton_end_points

    if len(end_points1) > len(end_points2):
        end_points1, end_points2 = end_points2, end_points1
        mas1, mas2 = mas2, mas1
        g_pts, gprime_pts = gprime_pts, g_pts

    k = len(end_points1)
    n = len(end_points2)

    corr = np.zeros((n, n))
    for i, ep1 in enumerate(mas1.skeleton_end_point_indices):
        for j, ep2 in enumerate(mas2.skeleton_end_point_indices):
            pd_mat = path_distance_matrix(ep1, mas1, ep2, mas2)
            corr[i, j] = optimal_subsequence_bijection(pd_mat)

    const = np.mean(corr)
    if n > k:
        corr[k:n, :] = const

    _, gp = linear_sum_assignment(np.array(corr))
    pairs = []
    total_cost = 0
    for i, j in enumerate(gp):
        if i < len(mas1.skeleton_end_point_indices):
            gi = mas1.skeleton_end_point_indices[i]
            gj = mas2.skeleton_end_point_indices[j]
            total_cost += corr[i, j]
            pairs.append((gi, gj))

    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    fig.set_size_inches((9.6, 7.2))

    ax[0, 0].scatter(end_points1[:, 0], end_points1[:, 1], c="b")
    ax[0, 0].axis('off')
    ax[0, 1].scatter(end_points2[:, 0], end_points2[:, 1], c="r")
    ax[0, 1].axis('off')
    gp1, gp2 = map(np.array, (g_pts, gprime_pts))
    ax[1, 0].scatter(gp1[:, 0], gp1[:, 1])
    ax[1, 0].axis('off')
    ax[1, 1].scatter(gp2[:, 0], gp2[:, 1])
    ax[1, 1].axis('off')

    for pt in end_points1:
        ax[0, 0].annotate(str(mas1.medial_axis_point_idx_map[tuple(pt)]), pt)

    for pt in end_points2:
        ax[0, 1].annotate(str(mas2.medial_axis_point_idx_map[tuple(pt)]), pt)

    end = timer()
    plt.suptitle(str(pairs))
    plt.show()
    print(f"Total Cost: {total_cost}")
    print(f"Time taken = {round(end - start, 2)}")


def select_ref_images():
    for file in Path("shapes").glob("*1.txt"):
        if "11" not in file.name:
            plt.rcParams["figure.figsize"] = (9.6, 7.2)
            plt.figure(num=file.stem)
            pts = extract_points_from_file(str(file))
            pts = np.array(pts)
            plt.scatter(pts[:, 0], pts[:, 1])
            plt.title(file.name)
            plt.gca().set_xlim([-1, 11])
            plt.gca().set_ylim([-1, 11])
            plt.show()


def match_shape(query_image="shapes/camel/camel_2.txt"):
    query_pts = extract_points_from_file(query_image)
    query_skeleton = MedialAxisTransformer.from_boundary_points(query_pts)

    match_score = {}

    ref_files = [file for file in Path(
        "reference_shapes/all_shapes").glob("*.txt")]
    ref_pts = list(map(extract_points_from_file, ref_files))
    ref_skeletons = list(map(
        MedialAxisTransformer.from_boundary_points, ref_pts))
    fargs = list(zip(ref_skeletons, [file.stem for file in ref_files]))

    for skel, skel_name in tqdm(fargs):
        try:
            img_class, score = compare_skeletons(
                query_skeleton, skel, skel_name)
        except Exception as e:
            e
            continue
        match_score[img_class] = score

    imclasses = list(match_score.keys())
    scores = np.array(list(match_score.values()))
    exp_scores = np.power(np.euler_gamma, scores)
    probs = exp_scores / np.sum(exp_scores)

    # best_match = imclasses[np.argmax(probs)]
    print(f"Best matches for {Path(query_image).stem}")

    for ic, pr in zip(imclasses, probs):
        match_score[ic] = round(pr * 100, 2)

    top_5_matches = {k: n for k, n in sorted(
        match_score.items(), key=lambda x: x[1], reverse=True)[:5]}

    print(json.dumps(top_5_matches, indent="\t"))


if __name__ == "__main__":

    # main2(if1, if2)
    match_shape()
