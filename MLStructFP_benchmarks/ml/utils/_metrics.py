"""
MLSTRUCT-FP BENCHMARKS - ML - UTILS - METRICS

Metric functions.
"""

__all__ = [
    'binary_accuracy_metric',
    'iou_metric',
    'r2_score_metric'
]

from typing import Union

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


def iou_metric(y_true: 'np.ndarray', y_pred: 'np.ndarray', threshold: Union[float, 'tf.Tensor'] = 0) -> float:
    """
    Compute IoU.

    :param y_true: True value matrix
    :param y_pred: Predicted matrix
    :param threshold: Minimum threshold to be considered 1. If zero, the threshold is not applied
    :return: Metric
    """
    results = []
    threshold = float(threshold)
    if 0 < threshold < 1:
        y_pred = np.where(y_pred > threshold, 1, 0)
    if len(y_true.shape) == len(y_pred.shape) == 3:
        y_true = y_true.reshape((1, *y_true.shape))
        y_pred = y_pred.reshape((1, *y_pred.shape))
    for i in range(0, y_true.shape[0]):
        intersect = np.sum(y_true[i, :, :] * y_pred[i, :, :])
        union = np.sum(y_true[i, :, :]) + np.sum(y_pred[i, :, :]) - intersect + 1e-7
        results.append(np.mean((intersect / union)).astype(np.float32))
    return float(np.mean(results))


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
