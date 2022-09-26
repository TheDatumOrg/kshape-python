# *k*-Shape: Efficient and Accurate Clustering of Time Series
<div align="center">
<p>
<img alt="GitHub" src="https://img.shields.io/github/license/TheDatumOrg/kshape-python"> <a href="https://badge.fury.io/py/kshape"><img src="https://badge.fury.io/py/kshape.svg" alt="PyPI version" height="20"></a> <img alt="GitHub issues" src="https://img.shields.io/github/issues/TheDatumOrg/kshape-python"> <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/kshape">
</p>
</div>

*k*-Shape is a highly accurate and efficient unsupervised method for univariate and multivariate time-series clustering. *k*-Shape appeared at the ***ACM SIGMOD 2015*** conference, where it was selected as one of the (2) ***best papers*** and received the inaugural ***2015 ACM SIGMOD Research Highlight Award***. An extended version appeared in the ***ACM TODS 2017*** journal. Since then, *k*-Shape has achieved state-of-the-art performance in both ***univariate*** and ***multivariate*** time-series datasets (i.e., *k*-Shape is among the fastest and most accurate time-series clustering methods, ranked in the top positions of established benchmarks with 100+ different datasets). 

*k*-Shape has been widely adopted across different scientific areas (e.g., computer science, social science, space science, engineering, econometrics, biology, neuroscience, and medicine), Fortune 100-500 enterprises (e.g., Exelon, Nokia, and many financial firms), and organizations such as the European Space Agency.

If you use *k*-Shape in your project or research, cite the following two papers:

* [ACM SIGMOD 2015](https://www.paparrizos.org/papers/PaparrizosSIGMOD2015.pdf)
* [ACM TODS 2017](https://www.paparrizos.org/papers/PaparrizosTODS2017.pdf)

## References

> "k-Shape: Efficient and Accurate Clustering of Time Series"<br/>
> John Paparrizos and Luis Gravano<br/>
> 2015 ACM SIGMOD International Conference on Management of Data (**ACM SIGMOD 2015**)<br/>

```bibtex
@inproceedings{paparrizos2015k,
  title={{k-Shape: Efficient and Accurate Clustering of Time Series}},
  author={Paparrizos, John and Gravano, Luis},
  booktitle={Proceedings of the 2015 ACM SIGMOD international conference on management of data},
  pages={1855--1870},
  year={2015}
}
```

> "Fast and Accurate Time-Series Clustering"<br/>
> John Paparrizos and Luis Gravano<br/>
> ACM Transactions on Database Systems (**ACM TODS 2017**), volume 42(2), pages 1-49<br/>

```bibtex
@article{paparrizos2017fast,
  title={{Fast and Accurate Time-Series Clustering}},
  author={Paparrizos, John and Gravano, Luis},
  journal={ACM Transactions on Database Systems (ACM TODS)},
  volume={42},
  number={2},
  pages={1--49},
  year={2017}
}
```

## Acknowledgements

We thank [Teja Bogireddy](https://github.com/bogireddytejareddy) for his valuable help on this repository.

We also thank the initial contributors [Jörg Thalheim](https://github.com/Mic92) and [Gregory Rehm](https://github.com/hahnicity). The initial code was used in [Sieve](https://sieve-microservices.github.io/).

# *k*-Shape's Python Repository

This repository contains the Python implementation for *k*-Shape. For the Matlab version, check [here](https://github.com/thedatumorg/kshape-matlab).

## Data

To ease reproducibility, we share our results over two established benchmarks:

* The UCR Univariate Archive, which contains 128 univariate time-series datasets. 
  * Download all 128 preprocessed datasets [here](https://www.thedatum.org/datasets/UCR2022_DATASETS.zip).
* The UAE Multivariate Archive, which contains 28 multivariate time-series datasets. 
  * Download the first 14 preprocessed datasets [here](https://www.thedatum.org/datasets/UAE2022_DATASETS_1.zip).
  * Download the remaining 14 preprocessed datasets [here](https://www.thedatum.org/datasets/UAE2022_DATASETS_2.zip).

For the preprocessing steps check [here](https://github.com/thedatumorg/UCRArchiveFixes).

## Installation

Our code has dependencies on the following python packages:

* [numpy](https://numpy.org/)
* [pytorch](https://pytorch.org/)

### Install from [pip](https://pypi.python.org/pypi/kshape)
```
$ pip install kshape
```

### Install from source
```
$ git clone https://github.com/thedatumorg/kshape-python
$ cd kshape-python
$ python setup.py install
```

## Benchmarking

We present the runtime performance of *k*-Shape when varying the number of time series, number of clusters, and the lengths of time series. (All results are the average of 5 runs.)

<p align="center">
    <img src="https://github.com/TheDatumOrg/kshape-python/blob/main/docs/benchmarks.png">
</p>

## Usage                                                                                                                                     

### Univariate Example:
```python
import numpy as np
from kshape.core import kshape as ks_cpu
from kshape.core_gpu import kshape as ks_gpu

univariate_ts_datasets = np.expand_dims(np.random.rand(200, 60), axis=2)
num_clusters = 3

# CPU Model
cpu_model = ks_cpu(univariate_ts_datasets, num_clusters, centroid_init='zero', max_iter=100)

labels = np.zeros(univariate_ts_datasets.shape[0])
for i in range(num_clusters):
    labels[cpu_model[i][1]] = i
    
cluster_centroids = np.zeros((num_clusters, univariate_ts_datasets.shape[1], univariate_ts_datasets.shape[2]))
for i in range(num_clusters):
    cluster_centroids[i] = cpu_model[i][0]
    
    
# GPU Model
gpu_model = ks_gpu(univariate_ts_datasets, num_clusters, centroid_init='zero', max_iter=100)

labels = np.zeros(univariate_ts_datasets.shape[0])
for i in range(num_clusters):
    labels[gpu_model[i][1]] = i
    
cluster_centroids = np.zeros((num_clusters, univariate_ts_datasets.shape[1], univariate_ts_datasets.shape[2]))
for i in range(num_clusters):
    cluster_centroids[i] = gpu_model[i][0].detach().cpu()
```

### Multivariate Example:
```python
import numpy as np
from kshape.core import kshape as ks_cpu
from kshape.core_gpu import kshape as ks_gpu

multivariate_ts_datasets = np.random.rand(200, 60, 6)
num_clusters = 3

# CPU Model
cpu_model = ks_cpu(multivariate_ts_datasets, num_clusters, centroid_init='zero', max_iter=100)

labels = np.zeros(multivariate_ts_datasets.shape[0])
for i in range(num_clusters):
    labels[cpu_model[i][1]] = i
    
cluster_centroids = np.zeros((num_clusters, multivariate_ts_datasets.shape[1], multivariate_ts_datasets.shape[2]))
for i in range(num_clusters):
    cluster_centroids[i] = cpu_model[i][0]
    
    
# GPU Model
gpu_model = ks_gpu(multivariate_ts_datasets, num_clusters, centroid_init='zero', max_iter=100)

labels = np.zeros(multivariate_ts_datasets.shape[0])
for i in range(num_clusters):
    labels[gpu_model[i][1]] = i
    
cluster_centroids = np.zeros((num_clusters, multivariate_ts_datasets.shape[1], multivariate_ts_datasets.shape[2]))
for i in range(num_clusters):
    cluster_centroids[i] = gpu_model[i][0].detach().cpu()
```

**Also see [Examples](https://github.com/TheDatumOrg/kshape-python/tree/master/examples) for UCR/UAE dataset clustering**               

## Results

The following tables contain the average Rand Index (RI), Adjusted Rand Index (ARI), and Normalized Mutual Information (NMI) accuracy values over 10 runs for *k*-Shape on the univariate and multivariate datasets.

Note: We collected the results using a single core implementation.

Server Specifications: AMD Ryzen 9 5900HX 8 Cores 3.30 GHz, 16GB RAM.

GPU Specifications: NVIDIA GeForce RTX 3070, 8GB memory.

### Univariate Results:

| Datasets              | RI | ARI | NMI     |  Runtime (secs) |
|:-----------------------:|:------------:|:------------:|:------------:|:-----------:|
| ACSF1               | 0.728889447  | 0.139127178  | 0.385362576  | 181.97282 |
| Adiac               | 0.948199219|0.237456072|0.585026777| 150.23389 |
| AllGestureWiimoteX               | 0.830988989|0.091833105|0.19967124| 132.64325 |
| AllGestureWiimoteY               |0.83356036|0.1306081|0.265320116| 68.32064 |
| AllGestureWiimoteZ               |0.831796196|0.08184644|0.184288361| 117.54415 |
| ArrowHead               |0.623696682|0.176408828|0.251716443| 1.42841 |
| Beef               |0.666553672|0.102291622|0.274983496| 2.04646 |
| BeetleFly               |0.518461538|0.037243262|0.049170634| 0.62138 |
| BirdChicken               |0.522948718|0.046863444|0.055805713| 0.46606 |
| BME               |0.623662322|0.209189215|0.337562447| 0.75734 |
| Car               |0.668095238|0.142785926|0.222574613| 4.87239 |
| CBF               |0.875577393|0.724563717|0.770334057| 7.47873 |
| Chinatown               |0.526075568|0.041117166|0.015693819| 0.548231 |
| ChlorineConcentration               |0.526233814|-0.001019087|0.000772354| 68.01957 |
| CinCECGTorso               |0.625307149|0.051803606|0.093350668| 271.74131 |
| Coffee               |0.726493506|0.453837834|0.421820948| 0.41349 |
| Computers               |0.529187976|0.058481715|0.0485609| 3.01130 |
| CricketX               |0.869701787|0.174655947|0.357916915| 55.23645 |
| CricketY               |0.873153945|0.206381317|0.373656368| 48.83094 |
| CricketZ               |0.869909812|0.172669605|0.355604411| 44.52660 |
| Crop               |0.924108349|0.241974335|0.4388123| 5420.01129 |
| DiatomSizeReduction               |0.919179195|0.807710845|0.827117298| 1.59904 |
| DistalPhalanxOutlineAgeGroup               |0.722184825|0.435943568|0.329905608| 2.12145 |
| DistalPhalanxOutlineCorrect               |0.499455708|-0.001030351|2.97E-05| 2.26317 |
| DistalPhalanxTW               |0.839607976|0.59272726|0.531060255| 10.96752 |
| DodgerLoopDay               |0.781988229|0.210916925|0.402897375| 1.69891 |
| DodgerLoopGame               |0.570071757|0.140620499|0.117161969| 0.86779 |
| DodgerLoopWeekend               |0.830807063|0.657966909|0.628131221| 0.495587 |
| Earthquakes               |0.541659908|0.024267193|0.006262268| 9.69413 |
| ECG200               |0.613753769|0.215794222|0.12870574| 0.74401 |
| ECG5000               |0.771307998|0.530703353|0.523220504| 163.82402 |
| ECGFiveDays               |0.811446734|0.623122565|0.586492573| 4.52766 |
| ElectricDevices               |0.693551963|0.071161449|0.177107461| 591.80007 |
| EOGHorizontalSignal               |0.86864851|0.227034804|0.408923026| 357.01975 |
| EOGVerticalSignal               |0.87082521|,0.200763231|0.37416983| 236.19376 |
| EthanolLevel               |0.622273617|0.003480205|0.007896876| 188.62335 |
| FaceAll               |0.910295025|0.433266026|0.610598916| 317.37956 |
| FaceFour               |0.757335907|0.374239896|0.466746543| 1.38740 |
| FacesUCR               |0.910295025|0.433266026|0.610598916| 136.62772 |
| FiftyWords               |0.951558207|0.358925864|0.651569015| 198.84656 |
| Fish               |0.785345886|0.189885615|0.327951361| 17.13432 |
| FordA               |0.564619244|0.129237686|0.096210429| 344.81591 |
| FordB               |0.516109383|0.032218211|0.023938345| 254.47971 |
| FreezerRegularTrain               |0.638744137|0.277488682|0.211547387| 18.45565 |
| FreezerSmallTrain               |0.639049682|0.278099783|0.212045663| 26.71921 |
| Fungi               |0.829126823|0.357543672|0.731173267| 6.11174 |
| GestureMidAirD1               |0.944819412|0.2937662|0.635503444| 30.88751 |
| GestureMidAirD2               |0.947697224|0.348582475|0.677310905| 43.38524 |
| GestureMidAirD3               |0.931266132|0.126759199|0.458782509|  18.98568 |
| GesturePebbleZ1               |0.883081466|0.585931482|0.675293127| 11.72848 |
| GesturePebbleZ2               |0.881353135|0.580554538|0.66392792| 7.60654 |
| GunPoint               |0.497487437|-0.005050505|0| 0.431333 |
| GunPointAgeSpan               |0.531991131|0.064141145|0.053146884| 1.59410 |
| GunPointMaleVersusFemale               |0.790127618|0.580242081|0.571776535| 1.08047 |
| GunPointOldVersusYoung               |0.518734664|0.037473134|0.028207614| 3.55970 |
| Ham               |0.528831556|0.057673104|0.044612673| 2.13764 |
| HandOutlines               |0.682856686|0.360051947|0.251176285| 247.46488 |
| Haptics               |0.689075575|0.063709939|0.09042192| 97.01234 |
| Herring               |0.501464075|0.003160642|0.007650463| 1.22652 |
| HouseTwenty               |0.520197437|0.040014774|0.03248788| 49.73466 |
| InlineSkate               |0.734065189|0.039846163|0.104643365| 372.13227 |
| InsectEPGRegularTrain               |0.706511773|0.363941816|0.379556522| 7.86684 |
| InsectEPGSmallTrain               |0.70409136|0.361370964|0.379504988| 5.37182 |
| InsectWingbeatSound               |0.792640539|0.196225831|0.402373638| 220.85374 |
| ItalyPowerDemand               |0.60972886|0.219608406|0.188152403| 3.01081 |
| LargeKitchenAppliances               |0.570070672|0.125576669|0.130422376| 12.03511 |
| Lightning2               |0.531294766|0.057017617|0.089783145| 1.93780 |
| Lightning7               |0.806175515|0.322963065|0.506494431| 4.51913 |
| Mallat               |0.924756461|0.721656055|0.869891088| 84.35894 |
| Meat               |0.761918768|0.494403401|0.580422751| 0.86227 |
| MedicalImages               |0.672005013|0.073490231|0.2287366| 32.23141 |
| MelbournePedestrian               |0.869441656|0.349104777|0.470402239| 275.40925 |
| MiddlePhalanxOutlineAgeGroup               |0.729585262|0.423115226|0.401722498| 1.57184 |
| MiddlePhalanxOutlineCorrect               |0.49977175|-0.00373634|0.000894849| 2.28809 |
| MiddlePhalanxTW               |0.809347564|0.449636118|0.431364361| 8.09901 |
| MixedShapesRegularTrain               |0.800991079|0.420414418|0.488448041| 285.77452 |
| MixedShapesSmallTrain               |0.800795029|0.419036374|0.4766379| 115.97755 |
| MoteStrain               |0.804809143|0.609589015|0.501865061| 4.56190 |
| NonInvasiveFetalECGThorax1               |0.950981974|0.33373922|0.676420909| 2995.88974 |
| NonInvasiveFetalECGThorax2               |0.967174335|0.465761156|0.765614776| 1748.11823 |
| OliveOil               |0.806892655|0.570012361|0.607418333| 1.97315 |
| OSULeaf               |0.785105837|0.263550973|0.361580708| 18.38517 |
| PhalangesOutlinesCorrect               |0.505362413|0.01070369|0.010221576| 6.79001 |
| Phoneme               |0.92769786|0.034705732|0.210108984| 1747.00270 |
| PickupGestureWiimoteZ               |0.854545455|0.288210152|0.540234358| 3.61598 |
| PigAirwayPressure               |0.903229862|0.03338252|0.427579631| 1632.92364 |
| PigArtPressure               |0.959821502|0.273442178|0.717389411| 914.99103 |
| PigCVP               |0.961346772|0.194516974|0.658363736| 1304.41961 |
| PLAID               |0.859444881|0.281634259|0.40487855| 555.89190 |
| Plane               |0.911765778|0.708344209|0.851592604| 1.14514 |
| PowerCons               |0.57637883|0.153069982|0.137929689| 1.74243 |
| ProximalPhalanxOutlineAgeGroup               |0.752674183|0.477154395|0.468537655| 1.72700 |
| ProximalPhalanxOutlineCorrect               |0.53390585|0.066453288|0.08535263| 1.15338 |
| ProximalPhalanxTW               |0.831222703|0.569454692|0.550694374| 5.31783 |
| RefrigerationDevices               |0.556208278|0.007595278|0.009437609| 28.19549 |
| Rock               |0.696935818|0.218081493|0.322230745| 179.14048 |
| ScreenType               |0.559603738|0.010528249|0.011742597| 26.81045 |
| SemgHandGenderCh2               |0.546315412|0.091559428|0.058471281| 39.87313 |
| SemgHandMovementCh2               |0.739443579|0.116429522|0.209097135| 195.28737 |
| SemgHandSubjectCh2               |0.724787047|0.19660949|0.263889093| 211.94098 |
| ShakeGestureWiimoteZ               |0.903171717|0.471533102|0.684959604| 3.51105 |
| ShapeletSim               |0.699939698|0.400050425|0.377331686| 3.14061 |
| ShapesAll               |0.978735474|0.42589872|0.742885495| 201.26739 |
| SmallKitchenAppliances               |0.398853939|0.004907405|0.02514159| 25.50886 |
| SmoothSubspace               |0.642434783|0.198252944|0.19954272| 2.06081 |
| SonyAIBORobotSurface1               |0.728057763|0.455518203|0.464021606| 2.53491 |
| SonyAIBORobotSurface2               |0.589140522|0.172496802|0.11750294| 4.86348 |
| StarLightCurves               |0.769194065|0.520688962|0.610221341| 64.50148 |
| Strawberry               |0.504165518|-0.019398783|0.123396507| 6.72441 |
| SwedishLeaf               |0.890254013|0.312306779|0.556179611| 58.87581 |
| Symbols               |0.880314418|0.619222941|0.757594317| 23.11830 |
| SyntheticControl               |0.881984975|0.600681896|0.712533175| 6.90626 |
| ToeSegmentation1               |0.50200682|0.004059369|0.005057191| 1.78287 |
| ToeSegmentation2               |0.635618839|0.260242738|0.191505717| 1.96561 |
| Trace               |0.711065327|0.455900994|0.598951999| 2.30357 |
| TwoLeadECG               |0.538024968|0.076155916|0.059000693| 8.53791 |
| TwoPatterns               |0.677979172|0.207830772|0.318418523| 185.70084 |
| UMD               |0.597057728|0.130992637|0.189184137| 0.93842 |
| UWaveGestureLibraryAll               |0.90364952|0.576024048|0.662693972| 288.38747 |
| UWaveGestureLibraryX               |0.85435587|0.353963525|0.457132359| 348.93967 |
| UWaveGestureLibraryY               |0.830476288|0.24845414|0.342123959| 471.75583 |
| UWaveGestureLibraryZ               |0.849091206|0.350080637|0.46397562| 448.39118 |
| Wafer               |0.541995609|0.026459678|0.010367784| 41.34034 |
| Wine               |0.496478296|-0.005187919|0.001056479| 0.57659 |
| WordSynonyms               |0.892537036|0.221578306|0.451754722| 74.17649 |
| Worms               |0.647528127|0.028458575|0.062591393| 24.33412 |
| WormsTwoClass               |0.503616566|0.00695446|0.009827969| 8.10779 |
| Yoga               | 0.499909412|-0.000340663|7.76E-05| 146.22124 |
 
### Multivariate Results:

| Datasets              | RI | ARI | NMI     |  Runtime (secs) |
|:-----------------------:|:------------:|:------------:|:------------:|:----------------:|
| ArticularyWordRecognition               |  0.766910468 | 0.070212062 | 0.311999115 | 729.28144 |
| AtrialFibrillation               | 0.572643678  | 0.021595081 | 0.091886209 | 50.86535 |
| BasicMotions               |  0.833575949 | 0.553263931 | 0.608454295 | 20.85916 |
| CharacterTrajectories               | 0.67428000  | 0.106482156 | 0.318298238 | 1476.40407 |
| Cricket               | 0.854357541  | 0.311436505 | 0.537425066 | 1005.15050 |
| DuckDuckGeese               | 0.628808080  | 0.019142319 | 0.080364497 | 8602.67186 |
| ERing               | 0.823360089  | 0.376640392 | 0.4555121907 | 44.22804 |
| Epilepsy               | 0.798144658  | 0.4714386229 | 0.5215828198 | 61.59339 |
| EthanolConcentration               |  0.541973056 | -0.001189755 | 0.002479593 | 134.94266 |
| FaceDetection               |  0.5000726770 | 0.000146178 | 0.000183457 | 3675.15437 |
| FingerMovements               | 0.503673540  | 0.0079117918 | 0.008676770 | 45.59016 |
| HandMovementDirection               |  0.5713180000 | -0.0001660879 | 0.0165459732 | 149.26488 |
| Handwriting               |  0.901074474 | 0.03098087 | 0.20089512 | 775.94464 |
| Heartbeat               | 0.500944436  | -0.003920272 | 0.001812514 | 641.62960 |
| InsectWingbeat               | 0.770585339  | 0.00250690814 | 0.00513089741 | 48609.98742 |
| JapaneseVowels               | 0.7119767214  | 0.0228757027 | 0.07046219420 | 111.86423 |
| LSST               | 0.782565525  | 0.04839431775 | 0.090181309 | 1337.49296 |
| Libras               | 0.870376044  | 0.17195485 | 0.416941213 | 94.94002 |
| MotorImagery               | 0.50023298  | 0.0007228376 | 0.0025653 | 1265.06028 |
| NATOPS               | 0.740513772  | 0.0877508740 | 0.14615484 | 93.30820 |
| PenDigits               | 0.8048295998  | 0.178763101 | 0.30736432 | 1598.41621 |
| PhonemeSpectra               | 0.945406005  | 0.019106322 | 0.11807663 | 15810.56167 |
| RacketSports               | 0.60430572  | 0.032991037 | 0.056093346 | 25.19951 |
| SelfRegulationSCP1               | 0.541451489  | 0.083061669 | 0.071136670 | 264.45256 |
| SelfRegulationSCP2               | 0.4989459797  | -0.00210444780 | 0.0003831136 | 446.17524 |
| SpokenArabicDigits               | 0.8100132143  | 0.1191198498 | 0.1892123 | 3266.35609 |
| StandWalkJump               | 0.575498575  | 0.11866387549 | 0.19928661087 | 62.83349 |
| UWaveGestureLibrary               | 0.7901325326  | 0.2077256357 | 0.358677650 | 202.24462 |
