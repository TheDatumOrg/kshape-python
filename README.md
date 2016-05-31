# k-Shape

Python implementation of [k-Shape](http://www.cs.columbia.edu/~jopa/kshape.html),
a new fast and accurate unsupervised Time Series cluster algorithm.
[See also](#relevant-articles)

## Usage

```
from kshape import kshape, zscore

time_series = [[1,2,3,4], [0,1,2,3], [0,1,2,3], [1,2,2,3]]
cluster_num = 2
clusters = kshape(zscore(time_series), cluster_num)
#=> [(array([-0.42860026, -1.15025211,  1.38751707, -0.42860026,  0.61993557]), [3]),
#    (array([-1.56839539, -0.40686255,  0.84042433,  0.67778452,  0.45704908]), [0, 1, 2])]
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

```
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

```
time_series = []
for f in df.columns:
  if not df[f].isnull().any() and df[f].var() != 0:
    time_series.append[df[f]]
```

## Relevant Articles

```
Paparrizos J and Gravano L (2015).
k-Shape: Efficient and Accurate Clustering of Time Series.
In Proceedings of the 2015 ACM SIGMOD International Conference on Management of Data, series SIGMOD '15,
pp. 1855-1870. ISBN 978-1-4503-2758-9, http://doi.org/10.1145/2723372.2737793. '
```
