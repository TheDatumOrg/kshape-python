from kshape.core import kshape, zscore

time_series = [[1,2,3,4,5], [0,1,2,3,4], [3,2,1,0,-1], [1,2,2,3,3]]
cluster_num = 2
clusters = kshape(zscore(time_series), cluster_num)
print(clusters)
