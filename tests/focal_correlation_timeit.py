from focal_stats import focal_correlation
import numpy as np

x = np.random.rand(10_000, 10_000)
y = np.random.rand(10_000, 10_000)

c = focal_correlation(x, y, verbose=True)
