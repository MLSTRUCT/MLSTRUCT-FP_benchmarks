"""
MLSTRUCT-FP BENCHMARKS - ML - MODEL - UTILS - PLOT

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
    from MLStructFP_benchmarks.ml.utils.plot._keras import *
else:
    plot_model_architecture = None

from MLStructFP_benchmarks.ml.utils.plot._plot_model import *
from MLStructFP_benchmarks.ml.utils.plot._utils import *
