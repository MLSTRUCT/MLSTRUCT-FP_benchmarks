"""
MLSTRUCTFP BENCHMARKS - ML - MODEL - ARCHITECTURES - BASE

Base FP model.
"""

__all__ = [
    'BaseFloorPhotoModel',
    'free',
    'iou',
    '_PATH_LOGS'
]

from abc import ABC

# noinspection PyProtectedMember
from MLStructFP_benchmarks.ml.model.core._model import GenericModel, _PATH_SESSION, _PATH_LOGS
from MLStructFP_benchmarks.ml.utils import iou_metric  # , jaccard_distance_loss

from typing import Tuple, TYPE_CHECKING, Any, Dict, Optional
import gc
import numpy as np
import os
import random
import time
import tensorflow as tf

if TYPE_CHECKING:
    from MLStructFP_benchmarks.ml.model.core import DataFloorPhoto

_IOU_THRESHOLD = 0.3


def free() -> None:
    """
    Free memory fun.
    """
    time.sleep(1)
    gc.collect()
    time.sleep(1)


def iou(y_true, y_pred):
    """
    IoU metric.

    :param y_true: True value matrix
    :param y_pred: Predicted matrix
    :return: Tf function to be used as a metric
    """
    return tf.py_function(iou_metric, [y_true, y_pred, _IOU_THRESHOLD], tf.float32)


class BaseFloorPhotoModel(GenericModel, ABC):
    """
    Base model image generation.
    """
    _data: 'DataFloorPhoto'
    _img_size: int
    _image_shape: Tuple[int, int, int]
    _samples: Dict[int, Dict[str, 'np.ndarray']]

    def __init__(
            self,
            data: Optional['DataFloorPhoto'],
            name: str,
            image_shape: Optional[Tuple[int, int, int]] = None,
            **kwargs
    ) -> None:
        """
        Constructor.

        :param data: Model data. Images must be between range (0, 1)
        :param name: Model name
        :param image_shape: Input shape
        :param kwargs: Optional keyword arguments
        """

        # Load data
        GenericModel.__init__(self, name=name, path=kwargs.get('path', ''))

        # Input shape
        if data is not None:
            assert data.__class__.__name__ == 'DataFloorPhoto', \
                f'Invalid data class <{data.__class__.__name__}>'
            self._data = data
            self._image_shape = data.get_image_shape()
            self._test_split = 1 - data.train_split
        else:
            assert image_shape is not None, 'If data is none, input_shape must be provided'
            assert isinstance(image_shape, tuple)
            assert len(image_shape) == 3
            assert image_shape[0] == image_shape[1]
            self._image_shape = image_shape

        self._samples = {}
        self._img_size = self._image_shape[0]
        self._info(f'Image shape {self._image_shape}')

    def _info(self, msg: str) -> None:
        """
        Information to console.

        :param msg: Message
        """
        if self._production:
            return
        self._print(f'{self._name}: {msg}')

    def reset_train(self) -> None:
        """
        Reset train.
        """
        super().reset_train()
        self._samples.clear()

    def predict_image(self, img: 'np.ndarray', threshold: bool = True) -> 'np.ndarray':
        """
        Predict image from common input.

        :param img: Image
        :param threshold: Use threshold
        :return: Image
        """
        if len(img) == 0:
            return img
        if len(img.shape) == 2:
            img = img.reshape(self._image_shape)
        if len(img.shape) == 3:
            img = img.reshape((-1, img.shape[0], img.shape[1], img.shape[2]))
        pred_img = self.predict(img)
        if threshold:
            pred_img = np.where(pred_img > _IOU_THRESHOLD, 1, 0)
        if len(pred_img.shape) == 4 and pred_img.shape[0] == 1:
            pred_img = pred_img.reshape((pred_img.shape[1], pred_img.shape[2], pred_img.shape[3]))
        return pred_img

    def _custom_save_session(self, filename: str, data: dict) -> None:
        """
        See upper doc.
        """
        # Save samples dict
        if len(self._samples.keys()) > 0:
            if self._get_session_data('train_samples') is None:
                self._register_session_data('train_samples', os.path.join(_PATH_SESSION, f'samples_{random.getrandbits(64)}.npz'))
            samples_f = self._get_session_data('train_samples')
            np.savez_compressed(samples_f, data=self._samples)

    def _custom_load_session(
            self,
            filename: str,
            asserts: bool,
            data: Dict[str, Any],
            check_hash: bool
    ) -> None:
        """
        See upper doc.
        """
        samples_f: str = self._get_session_data('train_samples')  # Samples File
        if samples_f is not None:
            self._samples = np.load(samples_f, allow_pickle=True)['data'].item()
