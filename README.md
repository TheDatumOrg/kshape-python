# k-Shape

[![Build Status](https://travis-ci.org/Mic92/kshape.svg?branch=master)](https://travis-ci.org/Mic92/kshape)

Python implementation of [k-Shape](http://www.cs.columbia.edu/~jopa/kshape.html),
a new fast and accurate unsupervised time-series cluster algorithm.
[See also](#relevant-articles)

We used this implementation for our paper: [Sieve: Actionable Insights from Monitored Metrics in Distributed Systems](https://sieve-microservices.github.io/)

## Installation

kshape is available on PyPI https://pypi.python.org/pypi/kshape

```console
$ pip install kshape
```

### Install from source

If you are using a [virtualenv](https://virtualenv.pypa.io/en/stable/) activate it. Otherwise you can install
into the system python

```console
$ python setup.py install
```

## Usage

```python
from kshape.core import kshape, zscore

time_series = [[1,2,3,4], [0,1,2,3], [0,1,2,3], [1,2,2,3]]
cluster_num = 2
clusters = kshape(zscore(time_series, axis=1), cluster_num)
#=> [(array([-1.161895  , -0.38729833,  0.38729833,  1.161895  ]), [0, 1, 2]),
#    (array([-1.22474487,  0.        ,  0.        ,  1.22474487]), [3])]
```

Returns list of tuples with the clusters found by kshape. The first value of the
tuple is zscore normalized centroid. The second value of the tuple is the index
of assigned series to this cluster.
The results can be examined by drawing graphs of the zscore normalized values
and the corresponding centroid.

## Gotchas when working with real-world time series

- If the data is available from different sources with same frequency but at different points in time, it needs to be aligned.
- In the following a tab seperated file is assumed, where each column is a different observation;
  gapps in columns happen, when only a certain value at this point in time was obtained.

```python
import pandas as pd
# assuming the time series are stored in a tab seperated file, where `time` is
# the name of the column containing the timestamp
df = pd.read_csv(filename, sep="\t", index_col='time', parse_dates=True)
# use a meaningful sample size depending on how the frequency of your time series:
# Higher is more accurate, but if series gets too long, the calculation gets cpu and memory intensive.
# Keeping the length below 2000 values is usually a good idea.
df = df.resample("500ms").mean()
df.interpolate(method="time", limit_direction="both", inplace=True)
df.fillna(method="bfill", inplace=True)
```

- kshape also expect no time series with a constant observation value or 'n/a'

```python
time_series = []
for f in df.columns:
  if not df[f].isnull().any() and df[f].var() != 0:
    time_series.append[df[f]]
```

## Relevant Articles

### Original paper

```plain
Paparrizos J and Gravano L (2015).
k-Shape: Efficient and Accurate Clustering of Time Series.
In Proceedings of the 2015 ACM SIGMOD International Conference on Management of Data, series SIGMOD '15,
pp. 1855-1870. ISBN 978-1-4503-2758-9, http://doi.org/10.1145/2723372.2737793. '
```

### Our paper where we used the python implementation
```bibtex
@article{sieve-middleware-2017,
  author       = {J{\"o}rg Thalheim, Antonio Rodrigues, Istemi Ekin Akkus, Pramod Bhatotia, Ruichuan Chen, Bimal Viswanath, Lei Jiao, Christof Fetzer},
  title        = {Sieve: Actionable Insights from Monitored Metrics in Distributed Systems}
  booktitle    = {Proceedings of Middleware Conference (Middleware)},
  year         = {2017},
}
```
