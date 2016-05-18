## k-Shape

Python implementation of [k-Shape](http://www.cs.columbia.edu/~jopa/kshape.html),
a new fast and accurate unsupervised Time Series cluster algorithm

### Usage

```
from kshape import kshape, zscore

time_series = [[1,2,3,4], [0,1,2,3], [-1,1,-1,1], [1,2,2,3]]
cluster_num = 2
clusters = kshape(zscore(time_series), cluster_num)
```
