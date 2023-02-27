"""
MLSTRUCTFP BENCHMARKS - ML - UTILS - METRICS

Metric functions.
"""

__all__ = [
    'binary_accuracy_metric',
    'r2_score_metric'
]

import numpy as np
import tensorflow as tf


def binary_accuracy_metric(y_true: 'np.ndarray', y_pred: 'np.ndarray') -> float:
    """
    Binary accuracy.

    :param y_true: True value matrix
    :param y_pred: Predicted matrix
    :return: Metric
    """
    assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)
    metric = tf.keras.metrics.BinaryAccuracy()
    metric.update_state(y_true, y_pred)
    r = metric.result().numpy()
    del metric
    return float(r)


def r2_score_metric(y_true: 'tf.Tensor', y_pred: 'tf.Tensor') -> float:
    """
    R² score metric.
    
    :param y_true: True value for prediction
    :param y_pred: Predicted value as y=Ƒ(x)
    :return: Metric
    """
    ssres = np.sum(np.square(y_true - y_pred))
    sstot = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - ssres / sstot
