"""
MLSTRUCT-FP BENCHMARKS - ML - MODEL - CORE

Core classes.
"""

# Check if keras exists
__keras = True

try:
    # noinspection PyUnresolvedReferences
    from keras import backend as k
except ModuleNotFoundError:
    __keras = False

if __keras:
    from MLStructFP_benchmarks.ml.model.core._model import *
    from MLStructFP_benchmarks.ml.model.core._utils import *

from MLStructFP_benchmarks.ml.model.core._data_floor_photo import *
