import numpy as np


def optimal_subsequence_bijection(matx, warpwin=float('inf'), qskip=float('inf'),
                                  tskip=float('inf'), jumpcost=None, all_vals=False):
    # m = len(t1)
    # n = len(t2)
    m, n = matx.shape

    # matx = np.zeros((m, n))

    # for r in range(m):
    #     matx[r, :] = np.square(t2 - t1[r])

    if jumpcost is None:
        # minimum value in each row
        rowise_min = np.min(matx, axis=1)
        # temp
        # columnwise_min = np.min(matx, axis=0)
        # jumpcost = min(np.std(rowise_min) + np.mean(rowise_min),
        #                np.std(columnwise_min) + np.mean(columnwise_min))
        jumpcost = np.std(rowise_min) + np.mean(rowise_min)

    # matxE = np.full((m+1, n), np.inf)
    # temp
    matxE = np.full((m + 1, n), np.inf)
    matxE[:m, :n] = matx
    # temp
    # matxE[m, n] = 0
    pathcost, indxrow, indxcol = find_path_DAG(
        matxE, warpwin, qskip, tskip, jumpcost)
    pathcost = pathcost / len(indxrow)

    if all_vals:
        return pathcost, indxrow, indxcol, jumpcost
    else:
        return pathcost


def find_path_DAG(matx, warpwin, qskip, tskip, jumpcost):
    m, n = matx.shape
    weights = np.full_like(matx, np.inf)
    camefromcol = np.full_like(matx, -1)
    camefromrow = np.full_like(matx, -1)

    weights[0, :] = matx[0, :]
    weights[:, 0] = matx[:, 0]

    for i in range(m - 1):
        for j in range(i, n - 1):
            if abs(i - j) <= warpwin:
                stoprowjump = min([m, i + qskip + 1])
                for rowjump in range(i + 1, stoprowjump):
                    stopk = min([n, j + tskip + 1])
                    for k in range(j + 1, stopk):
                        newweight = (weights[i, j]
                                     + matx[rowjump, k]
                                     + ((rowjump - i - 1) + (k - j - 1)) * jumpcost)
                        if weights[rowjump, k] > newweight:
                            weights[rowjump, k] = newweight
                            camefromcol[rowjump, k] = j
                            camefromrow[rowjump, k] = i

    pathcost = np.min(weights[m - 2, 1:])
    mincol = np.argmin(weights[m - 2, 1:])
    mincol = mincol + 1
    minrow = m - 2

    indxcol, indxrow = [], []

    while minrow >= 0 and mincol >= 0:
        indxcol = [mincol] + indxcol
        indxrow = [minrow] + indxrow
        mincoltemp = camefromcol[minrow, mincol]
        minrow = int(camefromrow[minrow, mincol])
        mincol = int(mincoltemp)

    return pathcost, indxrow, indxcol


if __name__ == "__main__":
    t1 = np.array([1, 2, 8, 12, 6, 8.5])
    t2 = np.array([1, 2, 9,  3, 3,  5.5, 9, 100, 23, 31])

    matx = [[3.00e+02, 2.35e+02, 1.79e+02, 8.70e-01, 3.02e+02, 1.88e+02,
            1.75e+02],
            [4.61e+00, 4.60e-01, 9.33e+00, 2.51e+02, 4.92e+00, 8.94e+00,
            1.14e+01],
            [2.32e+01, 7.95e+00, 3.39e+00, 1.82e+02, 2.41e+01, 6.00e-01,
            2.70e-01],
            [4.50e-01, 6.08e+00, 2.26e+01, 2.97e+02, 4.80e-01, 1.87e+01,
            2.28e+01],
            [4.80e-01, 5.33e+00, 2.11e+01, 2.92e+02, 5.50e-01, 1.72e+01,
            2.12e+01],
            [2.58e+01, 9.16e+00, 3.20e+00, 1.75e+02, 2.67e+01, 8.60e-01,
            9.00e-02],
            [2.44e+01, 6.07e+00, 3.00e-02, 1.82e+02, 2.53e+01, 3.89e+00,
            3.11e+00]]

    print(optimal_subsequence_bijection(np.array(matx)))
