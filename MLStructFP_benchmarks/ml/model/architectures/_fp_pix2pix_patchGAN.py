"""
MLSTRUCTFP BENCHMARKS - ML - MODEL - ARCHITECTURES - PIX2PIX PATCHGAN

Pix2Pix generation + patchGAN.
https://github.com/williamFalcon/pix2pix-keras
"""

__all__ = ['Pix2PixPatchGANFloorPhotoModel']

# noinspection PyProtectedMember
from MLStructFP_benchmarks.ml.model.core._model import GenericModel, _PATH_LOGS, _RUNTIME_METADATA_KEY, _PATH_CHECKPOINT
from MLStructFP_benchmarks.ml.utils import scale_array_to_range
from MLStructFP_benchmarks.ml.utils.plot.architectures import Pix2PixPatchGANFloorPhotoModelPlot

from keras.layers import Flatten, Input, Dropout, BatchNormalization, Reshape, \
    Conv2D, Concatenate, Layer, UpSampling2D, Lambda, Dense
# from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
import keras.backend as tf_backend
import tensorflow as tf

from typing import Tuple, TYPE_CHECKING, Generator, Any, List, Optional
import datetime
import gc
import numpy as np
import os
import time
import traceback

if TYPE_CHECKING:
    from ml.model.core import DataFloorPhoto

_DIR_ATOB: str = 'AtoB'
_DIR_BTOA: str = 'BtoA'


def _free() -> None:
    """
    Free memory fun.
    """
    time.sleep(1)
    gc.collect()
    time.sleep(1)


def dcgan(
        generator_model: 'Model',
        discriminator_model: 'Model',
        input_img_dim: Tuple[int, int, int],
        patch_dim: Tuple[int, int]
) -> 'Model':
    """
    Here we do the following:
    1. Generate an image with the generator
    2. break up the generated image into patches
    3. feed the patches to a discriminator to get the avg loss across all patches
        (i.e. is it fake or not)
    4. the DCGAN outputs the generated image and the loss

    This differs from standard GAN training in that we use patches of the image
    instead of the full image (although a patch size = img_size is basically the whole image)

    :param generator_model:
    :param discriminator_model:
    :param input_img_dim: Input image dimension
    :param patch_dim: Patch dimension
    :return: DCGAN model
    """
    generator_input = Input(shape=input_img_dim, name='DCGAN_input')

    h, w = input_img_dim[0], input_img_dim[1]
    ph, pw = patch_dim

    # generated image model from the generator
    generated_image = generator_model(generator_input)

    # chop the generated image into patches
    list_row_idx = [(i * ph, (i + 1) * ph) for i in range(int(h / ph))]
    list_col_idx = [(i * pw, (i + 1) * pw) for i in range(int(w / pw))]

    list_gen_patch = []
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            x_patch = Lambda(lambda z: z[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :],
                             output_shape=(ph, pw, input_img_dim[2]))(generated_image)
            list_gen_patch.append(x_patch)

    # measure loss from patches of the image (not the actual image)
    dcgan_output = discriminator_model(list_gen_patch)

    # actually turn into keras model
    dc_gan = Model(
        inputs=[generator_input],
        outputs=[generated_image, dcgan_output],
        name='DCGAN'
    )
    return dc_gan


def _patch_gan_discriminator(
        output_img_dim: Tuple[int, int, int],
        patch_dim: Tuple[int, int, int],
        nb_patches: int,
        num_filters_start: int,
        out_patch_dim: int,
        out_name: str
) -> 'Model':
    """
    Creates the generator according to the specs in the paper below.
    [https://arxiv.org/pdf/1611.07004v1.pdf][5. Appendix]

    PatchGAN only penalizes structure at the scale of patches. This
    discriminator tries to classify if each N x N patch in an
    image is real or fake. We run this discriminator convolutationally
    across the image, averaging all responses to provide
    the ultimate output of D.

    The discriminator has two parts. First part is the actual discriminator
    seconds part we make it a PatchGAN by running each image patch through the model,
    and then we average the responses

    Discriminator does the following:
    1. Runs many pieces of the image through the network
    2. Calculates the cost for each patch
    3. Returns the avg of the costs as the output of the network

    :param output_img_dim: Output dimension
    :param patch_dim: (width, height, channels) T
    :param nb_patches: Number of patches
    :param out_patch_dim: Dimension of out size
    :param out_name: Name of out layer
    :return: Model
    """
    # -------------------------------
    # DISCRIMINATOR
    # C64-C128-C256-C512-C512-C512 (for 256x256)
    # otherwise, it scales from 64
    # 1 layer block = Conv - BN - LeakyRelu
    # -------------------------------
    input_layer = Input(shape=patch_dim)

    # We have to build the discriminator dynamically because
    # the size of the disc patches is dynamic
    nb_conv = int(np.floor(np.log(output_img_dim[1]) / np.log(2)))
    filters_list = [num_filters_start * min(8, (2 ** i)) for i in range(nb_conv)]

    # CONV 1
    # Do first conv bc it is different from the rest
    # paper skips batch norm for first layer
    disc_out = Conv2D(num_filters_start, kernel_size=4, strides=2, padding='same', name='disc_conv_1')(input_layer)
    disc_out = LeakyReLU(alpha=0.2)(disc_out)

    # CONV 2 - CONV N
    # do the rest of the convs based on the sizes from the filters
    for i, filter_size in enumerate(filters_list[1:]):
        name = f'disc_conv_{i + 2}'

        disc_out = Conv2D(filter_size, kernel_size=4, padding='same', strides=2, name=name)(disc_out)
        disc_out = BatchNormalization(name=name + '_bn')(disc_out)
        disc_out = LeakyReLU(alpha=0.2)(disc_out)

    # ------------------------
    # BUILD PATCH GAN
    # this is where we evaluate the loss over each sublayer of the input
    # ------------------------
    return _generate_patch_gan_loss(
        last_disc_conv_layer=disc_out,
        patch_dim=patch_dim,
        input_layer=input_layer,
        nb_patches=nb_patches,
        out_patch_dim=out_patch_dim,
        out_name=out_name
    )


def _generate_patch_gan_loss(
        last_disc_conv_layer: 'Layer',
        patch_dim: Tuple[int, int, int],
        input_layer: Layer,
        nb_patches: int,
        out_patch_dim: int,
        out_name: str
) -> 'Model':
    """
    Generate patch gan LOSS.

    :param last_disc_conv_layer: Last convolutional layer
    :param patch_dim: Patch dimension
    :param input_layer: Input layer
    :param nb_patches: Number of patches
    :param out_patch_dim: Dimension of out size
    :param out_name: Name of out layer
    :return: Model
    """
    # generate a list of inputs for the different patches to the network
    list_input = [Input(shape=patch_dim, name=f'patch_gan_input_{i}') for i in range(nb_patches)]

    # get an activation
    x_flat = Flatten()(last_disc_conv_layer)
    x = Dense(2, activation='softmax', name='disc_dense')(x_flat)

    patch_gan = Model(inputs=[input_layer], outputs=[x, x_flat], name='patch_gan')

    # generate individual losses for each patch
    x = [patch_gan(patch)[0] for patch in list_input]
    x_mbd = [patch_gan(patch)[1] for patch in list_input]

    # merge layers if you have multiple patches (aka perceptual loss)
    if len(x) > 1:
        x = concatenate(x, name='merged_features')
    else:
        x = x[0]

    # merge mbd if needed
    # mbd = mini batch discrimination
    # https://arxiv.org/pdf/1606.03498.pdf
    if len(x_mbd) > 1:
        x_mbd = concatenate(x_mbd, name='merged_feature_mbd')
    else:
        x_mbd = x_mbd[0]

    num_kernels = 100
    dim_per_kernel = 5

    x_mbd = Dense(num_kernels * dim_per_kernel, use_bias=False)(x_mbd)

    x_mbd = Reshape((num_kernels, dim_per_kernel))(x_mbd)
    x_mbd = Lambda(_minb_disc, output_shape=_lambda_output)(x_mbd)
    x = concatenate([x, x_mbd])

    x_out = Dense(out_patch_dim, activation='softmax', name=out_name)(x)

    discriminator = Model(inputs=list_input, outputs=[x_out], name='Discriminator')
    return discriminator


def _lambda_output(input_shape: Tuple[int, int, int]) -> Tuple:
    """
    Lambda function output.

    :param input_shape: Input shape
    :return: Shape
    """
    return input_shape[:2]


def _minb_disc(x: 'tf.Tensor') -> 'tf.Tensor':
    """
    MinB.

    :param x: Input tensor
    :return: Loss
    """
    diffs = tf_backend.expand_dims(x, 3) - tf_backend.expand_dims(tf_backend.permute_dimensions(x, [1, 2, 0]), 0)
    abs_diffs = tf_backend.sum(tf_backend.abs(diffs), 2)
    x = tf_backend.sum(tf_backend.exp(-abs_diffs), 2)
    return x


def _num_patches(
        output_img_dim: Tuple[int, int, int],
        sub_patch_dim: Tuple[int, int]
) -> Tuple[int, Tuple[int, int, int]]:
    """
    Creates non-overlapping patches to feed to the PATCH GAN
    (Section 2.2.2 in paper)
    The paper provides 3 options.
    Pixel GAN = 1x1 patches (aka each pixel)
    PatchGAN = nxn patches (non-overlapping blocks of the image)
    ImageGAN = im_size x im_size (full image)

    Ex: 4x4 image with patch_size of 2 means 4 non-overlapping patches

    :param output_img_dim: (channels, w, h)
    :param sub_patch_dim: (w, h)
    :return: Number of patches and dimension (w, h, channels)
    """
    # num of non-overlapping patches
    nb_non_overlapping_patches = (output_img_dim[1] / sub_patch_dim[0]) * (output_img_dim[2] / sub_patch_dim[1])

    # dimensions for the patch discriminator
    patch_disc_img_dim = (sub_patch_dim[0], sub_patch_dim[1], output_img_dim[0])

    return int(nb_non_overlapping_patches), patch_disc_img_dim


def _extract_patches(images: 'np.ndarray', sub_patch_dim: Tuple[int, int]) -> List['np.ndarray']:
    """
    Cuts images into k subpatches
    Each kth cut as the kth patches for all images
    ex: input 3 images [im1, im2, im3]
    output [[im_1_patch_1, im_2_patch_1], ... , [im_n-1_patch_k, im_n_patch_k]]

    :param images: array of Images (num_images, im_height, im_width, im_channels)
    :param sub_patch_dim: (height, width) ex: (30, 30) subpatch dimensions
    :return: List of patches
    """
    im_height, im_width = images.shape[1], images.shape[2]  # [len, w, h, ch]
    patch_height, patch_width = sub_patch_dim

    # list out all xs  ex: 0, 29, 58, ...
    x_spots = range(0, im_width, patch_width)

    # list out all ys ex: 0, 29, 58
    y_spots = range(0, im_height, patch_height)
    all_patches = []

    for y in y_spots:
        for x in x_spots:
            # indexing here is cra
            # images[num_images, num_channels, width, height]
            # this says, cut a patch across all images at the same time with this width, height
            image_patches = images[:, y: y + patch_height, x: x + patch_width, :]
            all_patches.append(np.asarray(image_patches, dtype=np.float32))
    return all_patches


def _get_disc_batch(
        x_original_batch: 'np.ndarray',
        x_decoded_batch: 'np.ndarray',
        generator_model: 'Model',
        make_fake: bool,
        patch_dim: Tuple[int, int],
        label_smoothing: bool = False,
        label_flipping: float = 0
) -> Tuple[List['np.ndarray'], 'np.ndarray']:
    """
    Generate a batch of data.

    :param x_original_batch: Original Y batch data (A)
    :param x_decoded_batch: Original X decoded data (B)
    :param generator_model: Generator model
    :param make_fake: Make fake examples
    :param patch_dim: Patch dimension
    :param label_smoothing: Label smoothing
    :param label_flipping: Label flipping
    :return: Data
    """
    # Create x_disc: alternatively only generated or real images
    if make_fake:
        # generate fake image
        # Produce an output
        x_disc = generator_model.predict(x_decoded_batch)

        # each image will produce a 1x2 vector for the results (aka is fake or not)
        y_disc = np.zeros((x_disc.shape[0], 2), dtype=np.uint8)

        # sets all first entries to 1. AKA saying these are fake
        # these are fake images
        y_disc[:, 0] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    else:
        # generate real image
        x_disc = x_original_batch

        # each image will produce a 1x2 vector for the results (aka is fake or not)
        y_disc = np.zeros((x_disc.shape[0], 2), dtype=np.uint8)
        if label_smoothing:
            y_disc[:, 1] = np.random.uniform(low=0.9, high=1, size=y_disc.shape[0])
        else:
            # these are real images
            y_disc[:, 1] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    # Now extract patches form x_disc
    x_disc = _extract_patches(images=x_disc, sub_patch_dim=patch_dim)

    return x_disc, y_disc


class Pix2PixPatchGANFloorPhotoModel(GenericModel):
    """
    Pix2Pix floor photo model image generation. Modified version.
    """
    _data: 'DataFloorPhoto'
    _dir: str  # Direction
    _path_logs: str  # Stores path logs
    _train_current_part: int
    _train_date: str
    _x: 'np.ndarray'  # Loaded data # a
    _xy: str
    _y: 'np.ndarray'  # Loaded data # b

    # Model properties
    _out_d_patch: int
    _gf: int
    _df: int
    _sub_patch_dim: Tuple[int, int]

    # Image properties
    _image_shape: Tuple[int, int, int]
    _img_channels: int
    _img_size: int
    _nbatches: int

    # Models
    _generator: 'Model'
    _discriminator: 'Model'

    plot: 'Pix2PixPatchGANFloorPhotoModelPlot'

    def __init__(
            self,
            data: Optional['DataFloorPhoto'],
            name: str,
            xy: str,
            direction: str,
            image_shape: Optional[Tuple[int, int, int]] = None,
            **kwargs
    ) -> None:
        """
        Constructor.

        :param data: Model data
        :param name: Model name
        :param xy: Which data use, if "x" learn from Architectural pictures, "y" from Structure
        :param direction: AtoB, BtoA
        :param image_shape: Input shape
        :param kwargs: Optional keyword arguments
        """
        assert xy in ['x', 'y'], 'Invalid xy, use "x" or "y"'
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
        self._xy = xy
        self._img_rows = self._image_shape[0]
        self._img_cols = self._image_shape[1]
        self._channels = self._image_shape[2]
        self._path_logs = os.path.join(self._path, _PATH_LOGS)
        assert self._img_cols == self._img_rows
        assert self._channels >= 1

        self._info(f'Direction {self._dir}')
        self._info(f'Learning representation from {xy}')

        # Register constructor
        self._register_session_data('dir', direction)
        self._register_session_data('image_shape', self._image_shape)
        self._register_session_data('xy', xy)

        # Calculate output shape of D
        self._out_d_patch = 2  # It must be 2, [0, 1] or [1, 0]
        self._sub_patch_dim = (64, 64)

        # Number of filters in the first layer of G and D
        self._gf = 64
        self._df = 64
        self._info(f'Discriminator filters ({self._df}), generator filters ({self._gf})')
        self._register_session_data('df', self._df)
        self._register_session_data('gf', self._gf)

        # Build and compile the discriminator
        self._nb_patch_patches, self._patch_gan_dim = _num_patches((self._channels, self._img_cols, self._img_rows),
                                                                   self._sub_patch_dim)
        self._discriminator = self._build_discriminator()

        # disable training while we put it through the GAN
        self._discriminator.trainable = False

        # -------------------------
        # Construct Computational
        #   Graph of Generator
        # -------------------------

        # Define Optimizers
        opt_discriminator = Adam(lr=1E-4, epsilon=1e-08)
        opt_dcgan = Adam(lr=1E-4, epsilon=1e-08)
        # opt_discriminator = Adam(0.0002, 0.5)
        # opt_dcgan = Adam(0.0002, 0.5)
        loss_discriminator = 'binary_crossentropy'  # MSE does not work as outputs are [0,1] or [1,0]

        # Build the generator
        self._generator = self.build_generator()
        # self._generator.compile(loss='mae', optimizer=opt_discriminator)

        # Build model
        self._model = dcgan(
            generator_model=self._generator,
            discriminator_model=self._discriminator,
            input_img_dim=self._image_shape,
            patch_dim=self._sub_patch_dim
        )

        self.compile(
            optimizer=opt_dcgan,
            loss=['mae', loss_discriminator],
            loss_weights=[100, 1],  # Generator, Discriminator
            # metrics=None,
            as_list=True
        )
        self._check_compilation = False

        # Re enable discriminator
        self._discriminator.trainable = True
        self._discriminator.compile(
            loss=loss_discriminator,
            optimizer=opt_discriminator,
            metrics=['accuracy']
        )

        self.plot = Pix2PixPatchGANFloorPhotoModelPlot(self)

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
                dropout_rate: float = 0.0
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
        return _patch_gan_discriminator(
            output_img_dim=self._image_shape,
            patch_dim=self._patch_gan_dim,
            nb_patches=self._nb_patch_patches,
            num_filters_start=self._df,
            out_patch_dim=self._out_d_patch,
            out_name='out_' + self._output_layers[0]
        )

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
            if not self.train_batch(
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
        part_data = self._data.load_part(part=part, xy=self._xy, remove_null=True, shuffle=False)
        xtrain_img: 'np.ndarray' = part_data[self._xy + '_rect'].copy()  # Unscaled, from range (0,255)
        ytrain_img: 'np.ndarray' = part_data[self._xy + '_fphoto'].copy()  # Unscaled, from range (0, 255)
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
                'generator_loss': [],
            }

        t_time_0 = time.time()
        _train_msg: str = '\t[Epoch %d/%d] [Batch %d/%d] [D loss: %f, %f] [G loss: %f] time: %s'
        _error: bool = False
        try:
            for epoch in range(epochs):
                for batch_i, (imgs_A, imgs_B) in enumerate(
                        self._load_batch(batch_size=batch_size, scale_to_1=_scale_to_1, shuffle=shuffle)
                ):

                    # ---------------------
                    #  Train Discriminator
                    # ---------------------

                    # Train the discriminators (original images = real / generated = Fake)
                    self._discriminator.trainable = True
                    x_discriminator, y_discriminator = _get_disc_batch(
                        imgs_A,
                        imgs_B,
                        self._generator,
                        make_fake=True,
                        patch_dim=self._sub_patch_dim)
                    d_loss_disc_f = self._discriminator.train_on_batch(x_discriminator, y_discriminator)  # [1,0]
                    x_discriminator, y_discriminator = _get_disc_batch(
                        imgs_A,
                        imgs_B,
                        self._generator,
                        make_fake=False,
                        patch_dim=self._sub_patch_dim)  # real
                    d_loss_disc_t = self._discriminator.train_on_batch(x_discriminator, y_discriminator)  # [0,1]
                    d_loss_disc = [d_loss_disc_f[0], d_loss_disc_t[0]]  # [fake->1, real->1]
                    self._discriminator.trainable = False

                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Train the generators
                    valid = np.zeros((batch_size, self._out_d_patch), dtype=np.uint8)
                    valid[:, 1] = 1
                    g_loss = self._model.train_on_batch(imgs_B, [imgs_A, valid])

                    elapsed_time = datetime.datetime.now() - start_time

                    # Plot the progress
                    print(_train_msg % (epoch + 1, epochs, batch_i, self._nbatches, d_loss_disc[0], d_loss_disc[1],
                                        g_loss[0], elapsed_time),
                          end='\r')

                    # If at save interval => save generated image samples
                    if sample_interval > 0 and batch_i % sample_interval == 0:
                        self.plot.samples(epoch, batch_i, save=True)
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

    def predict(self, x: Any) -> Any:
        """
        Predict image.

        :param x: Image
        :return: Predicted image
        """
        img = self._generator.predict(scale_array_to_range(x, (-1, 1), 'float32'))
        return scale_array_to_range(img, (0, 255), 'float32')

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
