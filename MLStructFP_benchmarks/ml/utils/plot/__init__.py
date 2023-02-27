"""
MLSTRUCTFP BENCHMARKS - ML - MODEL - UTILS - PLOT

Plotting methods and classes.
"""

# Check if keras exists
__keras = True

try:
    # noinspection PyUnresolvedReferences
    from keras import backend as k
except ModuleNotFoundError:
    __keras = False

if __keras:
    from MLStructFP_benchmarks.ml.utils.plot._keras import plot_model_architecture
else:
    plot_model_architecture = None

from MLStructFP_benchmarks.ml.utils.plot._plot_data_xy import DataXYPlot
from MLStructFP_benchmarks.ml.utils.plot._plot_model import compare_metrics_from_csv

from MLStructFP_benchmarks.ml.utils.plot._utils import annotate_heatmap, heatmap
