# k-Shape

Python implementation of [k-Shape](http://www.cs.columbia.edu/~jopa/kshape.html),
a new fast and accurate unsupervised Time Series cluster algorithm

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
n/aand the corresponding centroid.

## Gotchas when working with real-world time series

- If the data is available from different sources with same frequency but at different points in time, it needs to be aligned.
- In the following a tab seperated file is assumed, where each column is a different observation;
  gapps in columns happen, when only a certain value at this point in time was obtained.

```
import pandas as pd
# assuming the time series are stored in a tab seperated file, where `time` is
# the name of the column containing the timestamp
df = pd.read_csv(filename, sep="\t", index_col='time', parse_dates=True)
df = df.fillna(method="bfill", limit=1e9)
# drop rows with the same time stamp
df = df.groupby(level=0).first()
```

- kshape also expect no time series with a constant observation value or 'n/a'

```
time_series = []
for f in df.columns:
  if not df[f].isnull().any() and df[f].var() != 0:
    time_series.append[df[f]]
```
