import rasterio as rio
from focal_stats import focal_correlation_base, focal_correlation
import numpy as np

with rio.open("data/tree_height.asc") as f:
    m1 = f.read(indexes=1, out_dtype=np.float64)
with rio.open("data/wtd.tif") as f:
    m2 = f.read(indexes=1, out_dtype=np.float64)

c1 = focal_correlation_base(m1, m2, verbose=True)
c2 = focal_correlation(m1, m2, verbose=True)

assert np.allclose(c1, c2, equal_nan=True)
