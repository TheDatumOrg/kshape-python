import math
import torch
import numpy as np
from sklearn.base import ClusterMixin, BaseEstimator

def zscore(a, axis=0, ddof=0):
    mns = a.mean(dim=axis)
    sstd = a.std(dim=axis, unbiased=(ddof == 1))
    if axis and mns.dim() < a.dim():
        x=(a - mns.unsqueeze(axis)).div(sstd.unsqueeze(axis))
        return x.masked_fill(torch.isnan(x), 0)
        #return torch.nan_to_num((a - mns.unsqueeze(axis)).div(sstd.unsqueeze(axis)))
    else:
        x=a.sub_(mns).div(sstd)
        return x.masked_fill(torch.isnan(x), 0)
        #return torch.nan_to_num(a.sub_(mns).div(sstd))


def roll_zeropad(a, shift, axis=None):
    if shift == 0:
        return a
    if abs(shift) > len(a):
        return torch.zeros_like(a)

    padding = torch.zeros(abs(shift), a.shape[1], device="cuda", dtype=torch.float32)
    if shift < 0:
        return torch.cat((a[abs(shift):], padding))
    else:
        return torch.cat((padding, a[:-shift]))


def _ncc_c_3dim(x, y):
    den = torch.norm(x, p=2, dim=(0, 1)) * torch.norm(y, p=2, dim=(0, 1))

    if den < 1e-9:
        den = torch.tensor(float("inf"), device="cuda", dtype=torch.float32)

    x_len = x.shape[0]
    fft_size = 1 << (2*x_len-1).bit_length()

    cc = torch.fft.ifft(torch.fft.fft(x, fft_size, dim=0) * torch.conj(torch.fft.fft(y, fft_size, dim=0)), dim=0)
    cc = torch.cat((cc[-(x_len-1):], cc[:x_len]), dim=0)

    return torch.div(torch.sum(torch.real(cc), dim=-1), den)


def _sbd(x, y):
    ncc = _ncc_c_3dim(x, y)
    idx = ncc.argmax().item()
    yshift = roll_zeropad(y, (idx + 1) - max(len(x), len(y)))

    return yshift


def _extract_shape(idx, x, j, cur_center):
    _a = []
    for i in range(len(idx)):
        if idx[i] == j:
            if torch.sum(cur_center)==0:
                opt_x = x[i]
            else:
                opt_x = _sbd(cur_center, x[i])
            _a.append(opt_x)
            
    if len(_a) == 0:
        indices = torch.randperm(x.shape[0])[:1]
        return torch.squeeze(x[indices].clone())
        #return torch.zeros((x.shape[1]))

    a = torch.stack(_a)
    
    columns = a.shape[1]
    y = zscore(a, axis=1, ddof=1)
    
    s = y[:, :, 0].transpose(0, 1).mm(y[:, :, 0])

    p = torch.empty((columns, columns), device="cuda", dtype=torch.float32)
    p.fill_(1.0 / columns)
    p = torch.eye(columns, device="cuda", dtype=torch.float32) - p

    m = p.mm(s).mm(p)
    _, vec = torch.linalg.eigh(m,UPLO='U')
    centroid = vec[:, -1]

    finddistance1 = torch.norm(a.sub(centroid.reshape((x.shape[1], 1))), 2, dim=(1, 2)).sum()
    finddistance2 = torch.norm(a.add(centroid.reshape((x.shape[1], 1))), 2, dim=(1, 2)).sum()

    if finddistance1 >= finddistance2:
        centroid.mul_(-1)

    return zscore(centroid, ddof=1)


def _kshape(x, k, centroid_init='zero', max_iter=100):
    m = x.shape[0]
    idx = torch.randint(0, k, (m,), dtype=torch.float32).to("cuda")
    if centroid_init == 'zero':
        centroids = torch.zeros(k, x.shape[1], x.shape[2], device="cuda", dtype=torch.float32)
    elif centroid_init == 'random':
        indices = torch.randperm(x.shape[0])[:k]
        centroids = x[indices].clone()
    distances = torch.empty(m, k, device="cuda")

    for it in range(max_iter):
        old_idx = idx
        for j in range(k):
            for d in range(x.shape[2]):
                centroids[j, :, d] = _extract_shape(idx, torch.unsqueeze(x[:, :, d], axis=2), j, torch.unsqueeze(centroids[j, :, d], axis=1))
                #centroids[j] = torch.unsqueeze(_extract_shape(idx, x, j, centroids[j]), dim=1)
            
        for i, ts in enumerate(x):
            for c, ct in enumerate(centroids):
                dist = 1 - _ncc_c_3dim(ts, ct).max()
                distances[i, c] = dist
        
        idx = distances.argmin(1)
        if torch.equal(old_idx, idx):
            break

    return idx, centroids
    
def kshape(x, k, centroid_init='zero', max_iter=100):
    x = torch.tensor(x, device="cuda", dtype=torch.float32)
    idx, centroids = _kshape(x, k, centroid_init=centroid_init, max_iter=max_iter)
    clusters = []
    for i, centroid in enumerate(centroids):
        series = []
        for j, val in enumerate(idx):
            if i == val:
                series.append(j)
        clusters.append((centroid, series))

    return clusters

class KShapeClusteringGPU(ClusterMixin,BaseEstimator):
    labels_= None
    centroids_ = None

    def __init__(self,n_clusters, centroid_init='zero', max_iter=100):
        self.n_clusters = n_clusters
        self.centroid_init = centroid_init
        self.max_iter = max_iter


    def fit(self,X,y=None):
        clusters = self._fit(X,self.n_clusters, self.centroid_init, self.max_iter)
        self.labels_ = np.zeros(X.shape[0])
        self.centroids_ =torch.zeros(self.n_clusters, X.shape[1], X.shape[2], device="cuda", dtype=torch.float32)
        for i in range(self.n_clusters):
            self.labels_[clusters[i][1]] = i
            self.centroids_[i]=clusters[i][0]
        return self

    def predict(self, X):
        labels, _ = self._predict(X,self.centroids_)
        return labels
        
    
    def _predict(self,x, centroids):
        x = torch.tensor(x, device="cuda", dtype=torch.float32)
        m = x.shape[0]
        k=len(centroids)
        idx = torch.randint(0, self.n_clusters, (m,), dtype=torch.float32).to("cuda")
        distances = torch.empty(m, self.n_clusters, device="cuda")
                
        for i, ts in enumerate(x):
            for c, ct in enumerate(centroids):
                dist = 1 - _ncc_c_3dim(ts, ct).max()
                distances[i, c] = dist
            
        idx = distances.argmin(1)
    
    
        return idx, centroids
    
    
    def _fit(self,x, k, centroid_init='zero', max_iter=100):
        x = torch.tensor(x, device="cuda", dtype=torch.float32)
        idx, centroids = _kshape(x, k, centroid_init=centroid_init, max_iter=max_iter)
        clusters = []
        for i, centroid in enumerate(centroids):
            series = []
            for j, val in enumerate(idx):
                if i == val:
                    series.append(j)
            clusters.append((centroid, series))
    
        return clusters



if __name__ == "__main__":
    import sys
    import doctest
    sys.exit(doctest.testmod()[0])
