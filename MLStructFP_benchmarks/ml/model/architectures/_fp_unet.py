"""
MLSTRUCTFP BENCHMARKS - ML - MODEL - ARCHITECTURES - UNET

U-Net model.
"""

__all__ = ['UNETFloorPhotoModel']

# noinspection PyProtectedMember
from MLStructFP_benchmarks.ml.model.core._model import GenericModel, _PATH_SESSION
from MLStructFP_benchmarks.ml.utils import scale_array_to_range, iou_metric  # , jaccard_distance_loss
from MLStructFP_benchmarks.ml.utils.plot.architectures import UNETFloorPhotoModelPlot

from keras.layers import Input, Dropout, concatenate, Conv2D, UpSampling2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam

from typing import List, Tuple, TYPE_CHECKING, Any, Dict, Optional
import datetime
import gc
import numpy as np
import os
import random
import time
import tensorflow as tf

if TYPE_CHECKING:
    from ml.model.core import DataFloorPhoto

_IOU_THRESHOLD = 0.3


def _free() -> None:
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


class UNETFloorPhotoModel(GenericModel):
    """
    UNET model image generation.
    """
    _data: 'DataFloorPhoto'
    _samples: Dict[int, Dict[str, 'np.ndarray']]  # Samples for each part

    # Train
    _current_train_date: str
    _current_train_part: int

    # Image properties
    _img_channels: int
    _img_size: int
    _image_shape: Tuple[int, int, int]

    plot: 'UNETFloorPhotoModelPlot'

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

        self._output_layers = ['out']

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

        # Register constructor
        self._register_session_data('image_shape', self._image_shape)

        inputs = Input(self._image_shape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid', name=self._output_layers[0])(conv9)  # To binary, aka, just b/w

        self._model = Model(inputs=inputs, outputs=conv10)
        self.compile(
            optimizer=Adam(lr=1e-4),
            loss='binary_crossentropy',  # jaccard_distance_loss(),
            metrics=[iou, 'accuracy']
        )
        self._check_compilation = False

        self.plot = UNETFloorPhotoModelPlot(self)

    def _info(self, msg: str) -> None:
        """
        Information to console.

        :param msg: Message
        """
        if self._production:
            return
        self._print(f'UNETFloorPhoto: {msg}')

    def reset_train(self) -> None:
        """
        Reset train.
        """
        super().reset_train()
        self._samples.clear()

    def train(
            self,
            epochs: int,
            batch_size: int,
            val_split: float,
            shuffle: bool = True,
            **kwargs
    ) -> None:
        """
        See upper doc.

        Optional parameters:
            - init_part     Initial parts
            - n_samples     Number of samples
            - n_parts       Number of parts to be processed, if -1 there will be no limits
        """
        # Get initial parts
        init_part = kwargs.get('init_part', 1)
        assert isinstance(init_part, int)
        print_enabled = self._print_enabled

        # The idea is to train using each part of the data, metrics will not be evaluated
        total_parts: int = self._data.total_parts
        assert 1 <= init_part <= total_parts, \
            f'Initial part <{init_part}> exceeds total parts <{total_parts}>'

        if self._is_trained:
            print(f'Resuming train, last processed part: {max(list(self._samples.keys()))}')

        # Get number of samples
        n_samples = kwargs.get('n_samples', 3)
        assert isinstance(n_samples, int)
        assert n_samples >= 0
        if n_samples > 0:
            print(f'Evaluation samples: {n_samples}')
        _free()

        # Get total parts to be processed
        n_parts = int(kwargs.get('n_parts', -1))
        assert isinstance(n_parts, int)
        assert total_parts - init_part >= n_parts >= 1 or n_parts == -1  # -1: no limits
        if n_parts != -1:
            print(f'Number of parts to be processed: {n_parts}')

        npt = 0  # Number of processed parts
        for i in range(total_parts):
            if i > 0:
                self._print_enabled = False
            part = i + 1
            if part < init_part:
                continue

            print(f'Loading data part {part}/{total_parts}')
            part_data = self._data.load_part(part=part, shuffle=True)
            xtrain_img: 'np.ndarray' = part_data['photo']
            ytrain_img: 'np.ndarray' = part_data['binary']

            # Crop data
            _free()

            # Make sample inputs
            sample_id = np.random.randint(0, len(xtrain_img), n_samples)
            sample_input = xtrain_img[sample_id]
            sample_real = ytrain_img[sample_id]

            self._samples[part] = {
                'input': sample_input,
                'real': sample_real,
            }
            self._current_train_part = part
            self._current_train_date = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

            super()._train(
                xtrain=xtrain_img,
                ytrain=ytrain_img,
                xtest=None,
                ytest=None,
                epochs=epochs,
                batch_size=batch_size,
                val_split=val_split,
                shuffle=shuffle,
                use_custom_fit=False,
                continue_train=self._is_trained,
                compute_metrics=False
            )

            _free()
            del part_data
            if not self._is_trained:
                print('Train failed, stopping')
                self._print_enabled = print_enabled
                return

            # Predict samples
            if n_samples > 0:
                sample_predicted = self.predict_image(sample_input)

                # Save samples
                self._samples[part] = {
                    'input': sample_input,
                    'real': sample_real,
                    'predicted': sample_predicted
                }
            self._model.reset_states()

            npt += 1
            if npt == n_parts:
                print(f'Reached number of parts to be processed ({n_parts}), train has finished')
                break

        # Restore print
        self._print_enabled = print_enabled

    def predict_image(self, img: 'np.ndarray') -> 'np.ndarray':
        """
        Predict image from common input.

        :param img: Image
        :return: Image
        """
        if len(img) == 0:
            return img
        if len(img.shape) == 3:
            img = img.reshape((-1, img.shape[0], img.shape[1], img.shape[2]))
        pred_img = self.predict(img)
        pred_img = np.where(pred_img > _IOU_THRESHOLD, 1, 0)
        if len(pred_img.shape) == 4 and pred_img.shape[0] == 1:
            pred_img = pred_img.reshape((pred_img.shape[1], pred_img.shape[2], pred_img.shape[3]))
        return pred_img

    def predict(self, x: 'np.ndarray') -> 'np.ndarray':
        """
        See upper doc.
        """
        return self._model_predict(x=self._format_tuple(scale_array_to_range(x, (0, 1), 'uint8'), 'np', 'x'))

    def evaluate(self, x: 'np.ndarray', y: Tuple['np.ndarray', 'np.ndarray']) -> List[float]:
        """
        See upper doc.
        """
        return self._model_evaluate(
            x=self._format_tuple(x, 'np', 'x'),
            y=self._format_tuple(y, ('np', 'np'), 'y')
        )

    def _custom_save_session(self, filename: str, data: dict) -> None:
        """
        See upper doc.
        """
        # Save samples dict
        if len(self._samples.keys()) > 0:
            if self._get_session_data('train_samples') is None:
                self._register_session_data('train_samples', _PATH_SESSION +
                                            os.path.sep + f'samples_{random.getrandbits(64)}.npz')
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
