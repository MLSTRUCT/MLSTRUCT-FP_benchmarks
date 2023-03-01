"""
MLSTRUCTFP BENCHMARKS - ML - UTILS

Utility functions.
"""

# Check if keras exists
__keras = True

try:
    # noinspection PyUnresolvedReferences
    from keras import backend as k
except ModuleNotFoundError:
    __keras = False

from MLStructFP_benchmarks.ml.utils._array import get_key_hash
from MLStructFP_benchmarks.ml.utils._file import *
from MLStructFP_benchmarks.ml.utils._math import *

if __keras:
    from MLStructFP_benchmarks.ml.utils._loss import *
    from MLStructFP_benchmarks.ml.utils._metrics import *
