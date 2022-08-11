[![Documentation Status](https://readthedocs.org/projects/focal-stats/badge/?version=latest)](https://focal-stats.readthedocs.io/en/latest/?badge=latest)

This module aims to provide focal statistics for python, that runs without the installation of extensive GIS packages. 
The only dependency is numpy. For more details see the documentation.

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