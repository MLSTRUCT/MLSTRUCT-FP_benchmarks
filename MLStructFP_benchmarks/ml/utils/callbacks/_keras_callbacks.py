"""
MLSTRUCT-FP BENCHMARKS - ML - MODEL - UTILS - CALLBACKS - KERAS

Overwrites keras default callbacks.
"""

__all__ = [
    'BaseLogger',
    'History',
    'TimeHistory'
]

from keras.callbacks import Callback
import time
from typing import List


# noinspection PyMissingTypeHints,PyAttributeOutsideInit,PyMissingOrEmptyDocstring,PyUnusedLocal
class BaseLogger(Callback):
    """
    Callback that accumulates epoch averages of metrics.

    This callback is automatically applied to every Keras model.

    # Arguments
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over an epoch.
            Metrics in this list will be logged as-is in `on_epoch_end`.
            All others will be averaged in `on_epoch_end`.
    """

    def __init__(self, stateful_metrics=None) -> None:
        super().__init__()
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

    def on_epoch_begin(self, epoch, logs=None):
        self.seen = 0
        self.totals = {}

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        self.seen += batch_size

        for k, v in logs.items():
            if k in self.stateful_metrics:
                self.totals[k] = v
            else:
                if k in self.totals:
                    self.totals[k] += v * batch_size
                else:
                    self.totals[k] = v * batch_size

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for k in self.params['metrics']:
                if k in self.totals:
                    # Make value available to next callbacks
                    if k in self.stateful_metrics:
                        logs[k] = self.totals[k]
                    else:
                        logs[k] = self.totals[k] / self.seen


# noinspection PyMissingTypeHints,PyAttributeOutsideInit,PyMissingOrEmptyDocstring,PyUnusedLocal
class History(Callback):
    """
    Callback that records events into a `History` object.

    This callback is automatically applied to
    every Keras model. The `History` object
    gets returned by the `fit` method of models.
    """

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


# noinspection PyMissingOrEmptyDocstring,PyUnusedLocal
class TimeHistory(Callback):
    """
    Records the time history for each epoch.
    """
    times: List[float]
    epoch_time_start: float

    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, batch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs=None):
        self.times.append(time.time() - self.epoch_time_start)
