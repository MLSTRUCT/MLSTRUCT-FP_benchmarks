"""
MLSTRUCTFP BENCHMARKS - ML - MODEL - ARCHITECTURES - PIX2PIX MODIFIED

Pix2Pix generation, modified version from:
https://github.com/eriklindernoren/Keras-GAN/tree/master/pix2pix
https://github.com/eriklindernoren/Keras-GAN/blob/master/pix2pix/pix2pix.py
"""

__all__ = ['Pix2PixFloorPhotoModModel']

# noinspection PyProtectedMember
from MLStructFP_benchmarks.ml.model.core._model import GenericModel, _PATH_LOGS, _RUNTIME_METADATA_KEY, _PATH_CHECKPOINT
from MLStructFP_benchmarks.ml.utils import scale_array_to_range
from MLStructFP_benchmarks.ml.utils.plot.architectures import Pix2PixFloorPhotoModelModPlot

from keras.layers import Input, Dropout, LeakyReLU, BatchNormalization, \
    Conv2D, Concatenate, Layer, UpSampling2D
from keras.models import Model
from tensorflow.keras.optimizers import Adam

from typing import Tuple, TYPE_CHECKING, Generator, Any, List, Optional
import cv2
import datetime
import gc
import numpy as np
import os
import time
import traceback

if TYPE_CHECKING:
    from ml.model.core._data_floor_photo_xy import DataFloorPhotoXY

_DIR_ATOB: str = 'AtoB'
_DIR_BTOA: str = 'BtoA'
_DISCRIMINATOR_LOSS: str = 'binary_crossentropy'  # 'binary_crossentropy'


def _free() -> None:
    """
    Free memory fun.
    """
    time.sleep(1)
    gc.collect()
    time.sleep(1)


class Pix2PixFloorPhotoModModel(GenericModel):
    """
    Pix2Pix floor photo model image generation. Modified version.
    """
    _data: 'DataFloorPhotoXY'
    _dir: str  # Direction
    _path_logs: str  # Stores path logs
    _train_current_part: int
    _train_date: str
    _x: 'np.ndarray'  # Loaded data # a
    _y: 'np.ndarray'  # Loaded data # b

    # Model properties
    _gf: int
    _df: int
    _disc_patch: Tuple[int, int, int]

    # Image properties
    _channels: int
    _image_shape: Tuple[int, int, int]
    _img_size: int
    _nbatches: int

    # Models
    _generator: 'Model'
    _discriminator: 'Model'

    plot: 'Pix2PixFloorPhotoModelModPlot'

    def __init__(
            self,
            data: Optional['DataFloorPhotoXY'],
            name: str,
            direction: str,
            image_shape: Optional[Tuple[int, int, int]] = None,
            **kwargs
    ) -> None:
        """
        Constructor.

        :param data: Model data
        :param name: Model name
        :param direction: AtoB, BtoA
        :param image_shape: Input shape
        :param kwargs: Optional keyword arguments
        """
        assert direction in [_DIR_ATOB, _DIR_BTOA], \
            f'Invalid direction, use "{_DIR_ATOB}" or "{_DIR_BTOA}"'

        # Load data
        GenericModel.__init__(self, name=name, path=kwargs.get('path', ''))

        self._output_layers = ['discriminator', 'generator']

        # Input shape
        if data is not None:
            assert data.__class__.__name__ == 'DataFloorPhotoXY', \
                f'Invalid data class <{data.__class__.__name__}>'
            self._data = data
            self._image_shape = data.get_image_shape()
        else:
            assert image_shape is not None, 'If data is none, input_shape must be provided'
            assert isinstance(image_shape, tuple)
            assert len(image_shape) == 3
            assert image_shape[0] == image_shape[1]
            self._image_shape = image_shape

        self._dir = direction
        self._img_rows = self._image_shape[0]
        self._img_cols = self._image_shape[1]
        self._channels = self._image_shape[2]
        self._path_logs = os.path.join(self._path, _PATH_LOGS)
        assert self._img_cols == self._img_rows
        assert self._channels >= 1

        self._info(f'Direction {self._dir}')

        # Register constructor
        self._register_session_data('dir', direction)
        self._register_session_data('image_shape', self._image_shape)

        # Calculate output shape of D (PatchGAN)
        patch = int(self._img_rows / 2 ** 4)
        self._disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self._gf = 64
        self._df = 128
        self._info(f'Discriminator filters ({self._df}), generator filters ({self._gf})')
        self._register_session_data('df', self._df)
        self._register_session_data('gf', self._gf)

        # Build and compile the discriminator
        self._discriminator = self._build_discriminator()

        # -------------------------
        # Construct Computational
        #   Graph of Generator
        # -------------------------

        # Build the generator
        self._generator = self.build_generator()

        # Input images and their conditioning images
        img_a = Input(shape=self._image_shape)
        img_b = Input(shape=self._image_shape)

        # By conditioning on B generate a fake version of A
        fake_a = self._generator(img_b)

        # For the combined model we will only train the generator
        self._discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self._discriminator([fake_a, img_b])

        # Compiler properties
        adam_lr = 0.00015
        adam_beta = 0.5

        self._model = Model(inputs=[img_a, img_b], outputs=[valid, fake_a])
        self.compile(
            optimizer=Adam(adam_lr, adam_beta),
            loss=['mse', 'mae'],
            loss_weights=[1, 1],  # Discriminator, Generator
            # metrics=None,
            as_list=True
        )
        self._check_compilation = False

        # Re enable discriminator
        self._discriminator.trainable = True
        self._discriminator.compile(
            loss='mse',
            optimizer=Adam(adam_lr, adam_beta),
            metrics=['accuracy']
        )

        self.plot = Pix2PixFloorPhotoModelModPlot(self)

    def enable_verbose(self) -> None:
        """
        See upper doc.
        """
        raise RuntimeError('Method disabled on current model')

    def enable_tensorboard(self) -> None:
        """
        See upper doc.
        """
        raise RuntimeError('Method disabled on current model')

    def enable_early_stopping(self, monitor: str = 'val_loss', patience: int = 50, mode: str = 'auto') -> None:
        """
        See upper doc.
        """
        raise RuntimeError('Method disabled on current model')

    def enable_model_checkpoint(self, monitor: str = 'val_loss', epochs: int = 25, mode: str = 'auto') -> None:
        """
        See upper doc.
        """
        raise RuntimeError('Method disabled on current model')

    def enable_reduce_lr_on_plateau(
            self,
            monitor: str = 'val_loss',
            factor: float = 0.1,
            patience: int = 25,
            mode: str = 'auto',
            min_delta: float = 0.0001,
            cooldown: int = 0,
            min_lr: float = 0
    ) -> None:
        """
        See upper doc.
        """
        raise RuntimeError('Method disabled on current model')

    def disable_tqdm(self, *args, **kwargs) -> None:
        """
        See upper doc.
        """
        raise RuntimeError('Method disabled on current model')

    def disable_csv_logger(self) -> None:
        """
        See upper doc.
        """
        raise RuntimeError('Method disabled on current model')

    def _info(self, msg: str) -> None:
        """
        Information to console.

        :param msg: Message
        """
        if self._production:
            return
        self._print(f'Pix2PixFloorPhotoMod: {msg}')

    def build_generator(self) -> 'Model':
        """
        U-Net Generator.

        :return: Generated model
        """

        def conv2d(
                layer_input: 'Layer',
                filters: int,
                f_size: int = 4,
                bn: bool = True
        ) -> 'Layer':
            """
            Layers used during downsampling. Convolution 2D + LeakyRelu + BN.

            :param layer_input: Input layer
            :param filters: Number of filters
            :param f_size: Number of kernel size
            :param bn: Use BN
            :return: Layer
            """
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(
                layer_input: 'Layer',
                skip_input: 'Layer',
                filters: int,
                f_size: int = 4,
                dropout_rate: float = 0
        ) -> 'Layer':
            """
            Deconvolutional layer.

            :param layer_input: Input layer
            :param skip_input: Skip input layer
            :param filters: Number of filters
            :param f_size: Filter size
            :param dropout_rate: Dropout rate
            :return: Layer
            """
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self._image_shape)

        # Downsampling
        d1 = conv2d(d0, self._gf, bn=False)
        d2 = conv2d(d1, self._gf * 2)
        d3 = conv2d(d2, self._gf * 4)
        d4 = conv2d(d3, self._gf * 8)
        d5 = conv2d(d4, self._gf * 8)
        d6 = conv2d(d5, self._gf * 8)
        d7 = conv2d(d6, self._gf * 8)

        # Upsampling
        u1 = deconv2d(d7, d6, self._gf * 8)
        u2 = deconv2d(u1, d5, self._gf * 8)
        u3 = deconv2d(u2, d4, self._gf * 8)
        u4 = deconv2d(u3, d3, self._gf * 4)
        u5 = deconv2d(u4, d2, self._gf * 2)
        u6 = deconv2d(u5, d1, self._gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self._channels, kernel_size=4, strides=1, padding='same', activation='tanh',
                            name='out_' + self._output_layers[1])(u7)

        return Model(inputs=d0, outputs=output_img, name='Generator')

    def _build_discriminator(self) -> 'Model':
        """
        Build discriminator model.

        :return: Model
        """

        def d_layer(
                layer_input: 'Layer',
                filters: int,
                f_size: int = 4,
                bn: bool = True
        ):
            """
            Discriminator layer.

            :param layer_input: Input layer
            :param filters: Number of filters
            :param f_size: Number of kernels
            :param bn: Use batch normalization
            :return: Layer
            """
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_a = Input(shape=self._image_shape)
        img_b = Input(shape=self._image_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate()([img_a, img_b])

        d1 = d_layer(combined_imgs, self._df, bn=False)
        d2 = d_layer(d1, self._df * 2)
        d3 = d_layer(d2, self._df * 4)
        d4 = d_layer(d3, self._df * 8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same', name='out_' + self._output_layers[0])(d4)

        return Model(inputs=[img_a, img_b], outputs=validity, name='Discriminator')

    def _load_batch(
            self,
            batch_size: int,
            scale_to_1: bool,
            shuffle: bool
    ) -> Generator[int, Tuple['np.ndarray', 'np.ndarray'], None]:
        """
        Generate batch of elements to train.

        :param batch_size: Batch size
        :param scale_to_1: Normalize to (-1, 1)
        :param shuffle: Shuffle data
        :return: Iterator of images as numpy ndarray
        """
        if shuffle:
            indices = np.arange(self._x.shape[0])  # Sort
            np.random.shuffle(indices)
            self._x = self._x[indices]
            self._y = self._y[indices]
        self._nbatches = int(int(len(self._x) / batch_size))
        for i in range(self._nbatches - 1):
            _i = int(i * batch_size)
            _j = int((i + 1) * batch_size)
            _x: 'np.ndarray' = self._x[_i:_j]
            _y: 'np.ndarray' = self._y[_i:_j]
            if scale_to_1:
                _x = scale_array_to_range(_x, (-1, 1), 'float32')
                _y = scale_array_to_range(_y, (-1, 1), 'float32')
            yield _x, _y

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
        """
        raise RuntimeError('Use .train_batch() instead')

    def train_all_parts(
            self,
            epochs: int,
            batch_size: int = 1,
            shuffle: bool = True,
            sample_interval: int = 50,
            part_from: int = 1,
            part_to: int = -1
    ) -> None:
        """
        Train model with all data parts.

        :param epochs: Number of epochs
        :param batch_size: Batch size
        :param shuffle: Shuffle data on each epoch
        :param sample_interval: Interval to plot image samples
        :param part_from: From part
        :param part_to: To part, if -1 train to last
        """
        self._train_date = datetime.datetime.today().strftime('%Y%m%d%H%M%S')  # Initial train date
        total_parts = self._data.get_total_parts()
        if part_to == -1:
            part_to = total_parts
        assert 1 <= part_from < part_to <= total_parts
        print(f'Total parts to be trained: {part_to - part_from - 1}')
        for i in range(total_parts):
            part: int = i + 1
            if part < part_from:
                continue
            if part > part_to:
                break
            if self.train_batch(
                    epochs=epochs,
                    batch_size=batch_size,
                    part=i + 1,
                    shuffle=shuffle,
                    sample_interval=sample_interval,
                    reset_train_date=False
            ):
                break
        _free()

    def train_batch(
            self,
            epochs: int,
            batch_size: int,
            part: int,
            shuffle: bool,
            sample_interval: int,
            **kwargs
    ) -> bool:
        """
        Train model by batch, this function does not use default .train() model.

        :param epochs: Number of epochs
        :param batch_size: Batch size
        :param part: Part number
        :param shuffle: Shuffle data on each epoch
        :param sample_interval: Interval to plot image samples
        :param kwargs: Optional keyword arguments
        :return: True if train was successful
        """
        assert epochs >= 1
        assert batch_size >= 1
        assert sample_interval >= 0
        start_time = datetime.datetime.now()

        train_date_curr: str = datetime.datetime.today().strftime('%Y/%m/%d %H:%M:%S')
        if kwargs.get('reset_train_date', True):
            self._train_date = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
        self._train_current_part = part

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self._disc_patch)
        fake = np.zeros((batch_size,) + self._disc_patch)

        total_parts = self._data.get_total_parts()
        assert 1 <= part <= total_parts
        _crop_len: int = 0  # Crop to size
        _scale_to_1: bool = True  # Crop to scale

        if _crop_len != 0:
            print(f'Cropping: {_crop_len} elements')
        if not _scale_to_1:
            print('Scale to (-1,1) is disabled')
        if not shuffle:
            print('Data is not shuffled')

        # Remove previous data
        if hasattr(self, '_x'):
            del self._x, self._y

        print(f'Loading data part {part}/{total_parts}', end='')
        part_data = self._data.load_part(part=part, xy='y', remove_null=True, shuffle=False)
        xtrain_img: 'np.ndarray' = part_data['y_rect'].copy()  # Unscaled, from range (0,255)
        ytrain_img: 'np.ndarray' = part_data['y_fphoto'].copy()  # Unscaled, from range (0, 255)
        del part_data
        _free()

        # Crop data
        if _crop_len != 0:
            _cr = min(_crop_len, len(xtrain_img))
            xtrain_img, ytrain_img = xtrain_img[0:_cr], ytrain_img[0:_cr]

        if self._dir == _DIR_ATOB:
            self._y = xtrain_img
            self._x = ytrain_img
        elif self._dir == _DIR_BTOA:
            self._x = xtrain_img
            self._y = ytrain_img
        print('')

        if not hasattr(self, '_history') or len(self._history.keys()) == 0:
            self._history = {
                'discriminator_loss': [],
                'discriminator_accuracy': [],
                'generator_loss': []
            }

        t_time_0 = time.time()
        _train_msg: str = '\t[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s'
        _error: bool = False
        try:
            for epoch in range(epochs):
                for batch_i, (imgs_A, imgs_B) in enumerate(
                        self._load_batch(batch_size=batch_size, scale_to_1=_scale_to_1, shuffle=shuffle)
                ):

                    # ---------------------
                    #  Train Discriminator
                    # ---------------------

                    # Condition on B and generate a translated version
                    fake_a = self._generator.predict(imgs_B)

                    # Train the discriminators (original images = real / generated = Fake)
                    self._discriminator.trainable = True
                    d_loss_real = self._discriminator.train_on_batch([imgs_A, imgs_B], valid)
                    d_loss_fake = self._discriminator.train_on_batch([fake_a, imgs_B], fake)
                    self._discriminator.trainable = False
                    d_loss_avg = 0.5 * np.add(d_loss_real, d_loss_fake)
                    d_loss = d_loss_avg[0]
                    d_accuracy = d_loss_avg[1]

                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Train the generators
                    g_loss = self._model.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

                    elapsed_time = datetime.datetime.now() - start_time

                    # Plot the progress
                    print(_train_msg % (epoch + 1, epochs, batch_i, self._nbatches, d_loss,
                                        100 * d_accuracy, g_loss[0], elapsed_time),
                          end='\r')

                    # If at save interval => save generated image samples
                    if sample_interval > 0 and batch_i % sample_interval == 0:
                        self.plot.samples(epoch, batch_i, save=True)
                        self._history['discriminator_loss'].append(float(d_loss))
                        self._history['discriminator_accuracy'].append(float(d_accuracy))
                        self._history['generator_loss'].append(float(g_loss[0]))

        except KeyboardInterrupt:
            print('')
            print('Process interrupted by user')
            _error = True
        except Exception as e:
            traceback.print_exc()
            self._print(f'Uncaught Exception: {e}')
            _error = True

        _free()
        self._is_trained = True
        train_time = int(time.time() - t_time_0)
        if not _error:
            print('')
            print('\t[Part {0}/{1}] finished in {0} seconds'.format(part, total_parts, train_time))

            # Store weights
            model_checkpoint_root = f'{_PATH_CHECKPOINT}{os.path.sep}{self._name_formatted}'
            model_checkpoint_path = model_checkpoint_root + '{2}{0}{2}{1}'.format(self._train_date, '', os.path.sep)
            if not os.path.isdir(model_checkpoint_root):
                os.mkdir(model_checkpoint_root)
            if not os.path.isdir(model_checkpoint_path):
                os.mkdir(model_checkpoint_path)
            weights_file = os.path.join(model_checkpoint_path, f'weights_part{part}.h5')
            print(f'\tSaving weights to file: {weights_file}')
            self._model.save_weights(weights_file)

        # Save train data
        tr_mk = list(self._train_metadata.keys())
        if 'continue_train_count' not in tr_mk:
            self._train_metadata['continue_train_count'] = 1
        else:
            self._train_metadata['continue_train_count'] += 1
        if 'train_time' not in tr_mk:
            self._train_metadata['train_time'] = train_time
        else:
            self._train_metadata['train_time'] += train_time
        if _RUNTIME_METADATA_KEY not in tr_mk:
            self._train_metadata[_RUNTIME_METADATA_KEY] = {}
        t_meta = {
            'batch_size': batch_size,
            'compute_metrics': False,
            'continue_train': True,
            'csv_logger_file': '',
            'custom_train_fit': True,
            'epochs': len(self._history['generator_loss']),
            'max_epochs': len(self._history['generator_loss']),
            'model_checkpoint_path': '',
            'shuffle': shuffle,
            'tensorboard_log': '',
            'train_date': train_date_curr,
            'train_shape_x': self._make_shape((self._x,)),
            'train_shape_y': self._make_shape((self._y,)),
            'validation_split': 0
        }
        for k in list(t_meta.keys()):
            self._train_metadata[k] = t_meta[k]
        return _error

    def _load_data_samples(self, n: int) -> Tuple['np.ndarray', 'np.ndarray']:
        """
        Load n number of samples of data in random.

        :param n: Number of samples
        :return: Data from x, y; (a, b)
        """
        sample_id = np.random.randint(0, len(self._x), n)
        return self._x[sample_id], self._y[sample_id]

    def predict(self, x: 'np.ndarray') -> 'np.ndarray':
        """
        Predict image.

        :param x: Image
        :return: Predicted image
        """
        if len(x.shape) == 2:
            x = x.reshape((-1, x.shape[0], x.shape[1], 1))
        elif len(x.shape) == 3:
            x = x.reshape((-1, x.shape[0], x.shape[1], x.shape[2]))
        img = self._generator.predict(scale_array_to_range(x, (-1, 1), 'float32'))
        return scale_array_to_range(img, (0, 255), 'float32')

    def transform_input(self, x: 'np.ndarray') -> 'np.ndarray':
        """
        Transform input images of any shape to valid input image.

        :param x: Image numpy array
        :return: Transformed input
        """
        if len(x.shape) == 2:
            x = x.reshape((-1, x.shape[0], x.shape[1], 1))
        elif len(x.shape) == 3:
            x = x.reshape((-1, x.shape[0], x.shape[1], x.shape[2]))
        transformed: List['np.ndarray'] = []
        for i in range(len(x)):
            xi: 'np.ndarray' = x[i]

            # Resize
            if xi.shape[0] != self._image_shape[0] or xi.shape[1] != self._image_shape[1]:
                xi = cv2.resize(
                    src=xi,
                    dsize=(self._image_shape[0], self._image_shape[0]),
                    interpolation=cv2.INTER_AREA
                )

            # Channels
            if len(xi.shape) == 2 or xi.shape[2] == 1:
                xi = xi.reshape((xi.shape[0], xi.shape[1]))
                xi = np.stack((xi,) * self._channels, axis=-1)
            transformed.append(xi)
        return np.array(transformed)

    def get_xy(self, xy: str) -> Any:
        """
        See upper doc.
        """
        raise RuntimeError('Method disabled on current model')

    def evaluate(self, x: Any, y: Any) -> List[float]:
        """
        See upper doc.
        """
        raise RuntimeError('Method disabled on current model')

    def get_image_size(self) -> Tuple[int, int, int]:
        """
        Get image size (width, height, channels).

        :return: Image shape
        """
        return self._image_shape
