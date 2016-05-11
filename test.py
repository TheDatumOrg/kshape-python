#import pandas as pn
import numpy as np
from numpy.fft import fft, ifft
from scipy.sparse.linalg import eigs
import math
from scipy.stats import zscore
from scipy.ndimage.interpolation import shift


def next_greater_power_of_2(x):
    return 2**(x-1).bit_length()


def sbd(x, y):
    fft_size = next_greater_power_of_2(len(x))
    cc = np.abs(ifft(fft(x, fft_size) * fft(y, fft_size)))
    ncc = cc / math.sqrt((sum(x**2) * sum(y**2)))
    idx = ncc.argmax()
    dist = 1 - ncc[idx]
    return dist, shift(y, idx - len(x))


def extract_shape(x, c):
    n = len(x)
    m = len(x[0])
    new_x = np.zeros((n, m))
    for i, row in enumerate(x):
        _, x_i = sbd(c, row)
        new_x[i] = x_i
    s = np.dot(new_x.transpose(), new_x)
    q = np.identiy(len(s)) - np.ones(len(s)) * 1 / m
    M = np.dot(np.dot(q.transpose(), s), q)
    _, vec = eigs(M, 1)
    return vec[0]


def k_shape(x, k):
    iter_ = 0
    idx = np.zeros(len(x))  # TODO dimension, random init
    old_idx = np.zeros(len(x))
    c = np.zeros((k,)) # TODO dimension
    while idx != old_idx and iter_ < 100:
        old_idx = idx.copy()
        for j in range(k):
            x_ = []
            for i, x_i in enumerate(x):
                if idx(i) == j:
                    x_.append(x_i)
            c[j] = extract_shape(x_, c[j])
        for i, x_i in enumerate(x):
            min_dist = np.inf
            for j, c_j in enumerate(c):
                dist, _ = sbd(c_j, x_i)
            if dist < min_dist:
                min_dist = dist
                idx[i] = j


def test_extract_shape():
    a = zscore(np.ones((3, 10)))
    c = np.arange(10)
    extract_shape(a, c)


def test_sbd():
    a = np.arange(100)
    r1 = sbd(zscore(a), zscore(a))
    r2 = sbd(zscore(a), zscore(shift(a, 3)))
    r3 = sbd(zscore(a), zscore(shift(a, -3)))
    r4 = sbd(zscore(a), zscore(shift(a, 30)))
    r5 = sbd(zscore(a), zscore(shift(a, -30)))

if __name__ == "__main__":
    test_sbd()
    test_extract_shape()
