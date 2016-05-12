import math
import numpy as np

from numpy.random import randint
from numpy.linalg import norm
from numpy.fft import fft, ifft

from scipy.sparse.linalg import eigs
from scipy.stats import zscore
from scipy.ndimage.interpolation import shift

def _ncc_c(x,y):
    """
    >>> ncc_c([1,2,3,4], [1,2,3,4])
    array([ 0.13333333,  0.36666667,  0.66666667,  1.        ,  0.66666667,
            0.36666667,  0.13333333])
    >>> ncc_c([1,1,1], [1,1,1])
    array([ 0.33333333,  0.66666667,  1.        ,  0.66666667,  0.33333333])
    >>> ncc_c([1,2,3], [-1,-1,-1])
    array([-0.15430335, -0.46291005, -0.9258201 , -0.77151675, -0.46291005])
    """
    x_len = len(x)
    fft_size = 1<<(2*x_len-1).bit_length()
    cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size)))
    cc = np.concatenate((cc[-(x_len-1):], cc[:x_len]))
    return np.real(cc) / (norm(x) * norm(y))


def _sbd(x, y):
    """
    >>> sbd([1,1,1], [1,1,1])
    (-2.2204460492503131e-16, array([1, 1, 1]))
    >>> sbd([0,1,2], [1,2,3])
    (0.043817112532485103, array([1, 2, 3]))
    >>> sbd([1,2,3], [0,1,2])
    (0.043817112532485103, array([0, 1, 2]))
    """
    ncc = _ncc_c(x, y)
    idx = ncc.argmax()
    dist = 1 - ncc[idx]
    yshift = shift(y, (idx + 1) - max(len(x), len(y)))
    return dist, yshift

def _extract_shape(idx, x, j, cur_center):
    """
    >>> extract_shape(np.array([0,1,2]), np.array([[1,2,3], [4,5,6]]), 1, np.array([0,3,4]))
    array([ -1.00000000e+00,  -3.06658683e-19,   1.00000000e+00])
    >>> extract_shape(np.array([0,1,2]), np.array([[-1,2,3], [4,-5,6]]), 1, np.array([0,3,4]))
    array([-0.96836405,  1.02888681, -0.06052275])
    """
    _a = []
    for i in range(len(idx)):
        if idx[i] == j:
            if cur_center.sum() == 0:
                opt_x = x[i]
            else:
                _, opt_x = _sbd(cur_center, x[i])
            _a.append(opt_x)
    a = np.array(_a)

    if len(a) == 0:
        return np.zeros((1, x.shape[1]))
    columns = a.shape[1]
    y = zscore(a,axis=1,ddof=1)
    s = np.dot(y.transpose(), y)

    p = np.empty((columns, columns))
    p.fill(1.0/columns)
    p = np.eye(columns) - p

    m = np.dot(np.dot(p, s), p)
    _, vec = eigs(m, 1)
    centroid = np.real(vec[:,0])
    finddistance1 = math.sqrt(((a[0] - centroid) ** 2).sum())
    finddistance2 = math.sqrt(((a[0] + centroid) ** 2).sum())

    if finddistance1 >= finddistance2:
        centroid *= -1
    return zscore(centroid, ddof=1)


def _kshape(x, k):
    """
    >>> kshape(np.array([[1,2,3,4], [0,1,2,3], [-1,1,-1,1], [1,2,2,3]]), 2)
    (array([0, 0, 1, 0]), array([[-1.19623139, -0.26273649,  0.26273649,  1.19623139],
           [-0.8660254 ,  0.8660254 , -0.8660254 ,  0.8660254 ]]))
    """
    m = x.shape[0]
    idx = randint(0, k, size=m)
    centroids = np.zeros((k,x.shape[1]))

    distances = np.empty((m, k))
    for _ in range(100):
        old_idx = idx
        for j in range(k):
            res = _extract_shape(idx, x, j, centroids[j])
            centroids[j] = res

        for i in range(m):
            for j in range(k):
                distances[i,j] = 1 - max(_ncc_c(x[i], centroids[j]))
        idx = distances.argmin(1)
        if norm(old_idx - idx) == 0:
            break
    return idx, centroids

def kshape(x, k):
    idx, centroids = _kshape(np.array(x), k)
    clusters = []
    for i, centroid in enumerate(centroids):
        series = []
        for j, val in enumerate(idx):
            if i == val:
                series.append(j)
        clusters.append((centroid, series))
    return clusters

if __name__ == "__main__":
    import doctest
    doctest.testmod()
