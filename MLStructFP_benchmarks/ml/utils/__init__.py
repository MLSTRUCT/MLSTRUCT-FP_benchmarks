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

from MLStructFP_benchmarks.ml.utils._file import file_md5, load_history_from_csv

if __keras:
    from MLStructFP_benchmarks.ml.utils._loss import binary_cross_entropy, weighted_cross_entropy, \
        balanced_cross_entropy, focal_loss, dice_loss, jaccard_distance_loss, weighted_categorical_crossentropy, \
        pixelwise_softmax_crossentropy

    from MLStructFP_benchmarks.ml.utils._metrics import binary_accuracy_metric, r2_score_metric

from MLStructFP_benchmarks.ml.utils._math import filter_xylim, scale_array_to_range, r2_score
