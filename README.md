[![Documentation Status](https://readthedocs.org/projects/focal-stats/badge/?version=latest)](https://focal-stats.readthedocs.io/en/latest/?badge=latest)

This module aims to provide focal statistics for python, that runs without the installation of extensive GIS packages. 
It is mainly based on numpy. For more details see the documentation.

The package implements three different categories of spatial statistics:
- focal statistics, which are calculated as a moving window over input rasters
- grouped statistics, which calculates the statistics based on strata
- strata statistics, which calculates the statistics for each stratum and reapplies it to the input raster

Furthermore, this package provides functionality to implement your own focal statistics functions, based on the 
mechanisms of a rolling window.

# Installation

The package can be installed with conda:

```
conda install --channel conda-forge focal-stats
```

# Usage example

Focal mean of a 100x100 random numpy array.

```
from focal_stats import focal_mean
import numpy as np

x = np.random.rand(100, 100)
fm = focal_mean(x, window_size=5)
```

# Important links

- API reference: https://focal-stats.readthedocs.io/en/latest/api.html
- Documentation: https://focal-stats.readthedocs.io/en/latest/index.html