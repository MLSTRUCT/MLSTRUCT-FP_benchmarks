"""
MLSTRUCT-FP BENCHMARKS - ML - MODEL - ARCHITECTURES - PIX2PIX

Pix2Pix generation.
"""

__all__ = ['Pix2PixFloorPhotoModel']

from MLStructFP_benchmarks.ml.model.architectures._fp_base import *
from MLStructFP_benchmarks.ml.utils import scale_array_to_range
from MLStructFP_benchmarks.ml.utils.plot.architectures import Pix2PixFloorPhotoModelPlot
from MLStructFP.utils import DEFAULT_PLOT_DPI, DEFAULT_PLOT_STYLE

from keras.initializers import RandomNormal
from keras.layers import Input, Dropout, LeakyReLU, BatchNormalization, \
    Conv2D, Concatenate, Layer, Activation, Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam

from typing import List, Tuple, Union, TYPE_CHECKING, Optional
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

if TYPE_CHECKING:
    from MLStructFP_benchmarks.ml.model.core import DataFloorPhoto


class Pix2PixFloorPhotoModel(BaseFloorPhotoModel):
    """
    Pix2Pix floor photo model image generation.
    """
    _current_train_date: str
    _current_train_part: int

    # Models
    _d_model: 'Model'
    _g_model: 'Model'
    _patch: int

    plot: 'Pix2PixFloorPhotoModelPlot'

    def __init__(
            self,
            data: Optional['DataFloorPhoto'],
            name: str,
            image_shape: Optional[Tuple[int, int, int]] = None,
            **kwargs
    ) -> None:
        """
        Constructor.

        :param data: Model data
        :param name: Model name
        :param image_shape: Input shape
        :param kwargs: Optional keyword arguments
        """
        # Create base model
        # noinspection PyArgumentList
        BaseFloorPhotoModel.__init__(self, data, name, image_shape, **kwargs.get('path', ''))
        self._output_layers = ['discriminator', 'generator']

        # Register constructor data
        self._register_session_data('image_shape', self._image_shape)

        # Number of filters in the first layer of G and D
        df: int = 64
        gf: int = 64

        self._current_train_date: str = ''
        self._current_train_part: int = -1
        self._info(f'Discriminator filters ({df}), generator filters ({gf})')
        self._register_session_data('df', df)
        self._register_session_data('gf', gf)

        # Create models
        self._d_model = self._define_discriminator(
            input_shape=self._image_shape,
            output_shape=self._image_shape,
            df=df
        )
        self._g_model = self._define_generator(
            input_shape=self._image_shape,
            output_shape=self._image_shape,
            gf=gf
        )

        # Make d model not trainable while we put into GAN
        self._d_model.trainable = False

        # Define the source image
        in_src = Input(shape=self._image_shape, name='source_image')

        # Connect the source image to the generator input
        gen_out: 'Layer' = self._g_model(in_src)

        # Connect the source input and generator output to the discriminator input
        # Discriminators determines validity of translated images / condition pairs
        dis_out: 'Layer' = self._d_model([in_src, gen_out])

        # Src image as input, generated image and classification output
        self._model = Model(inputs=in_src, outputs=[dis_out, gen_out], name=self.get_name())

        # Compile the model
        self.compile(
            optimizer=Adam(lr=0.0002, beta_1=0.5),
            loss={
                self._output_layers[0]: 'binary_crossentropy',  # Discriminator
                self._output_layers[1]: 'mae'  # Generator
            },
            loss_weights={
                self._output_layers[0]: 1,  # Discriminator
                self._output_layers[1]: 100  # Generator
            },
            metrics={
                self._output_layers[0]: None,  # Discriminator
                self._output_layers[1]: [iou, 'accuracy']  # Generator
            }
        )
        self._check_compilation = False

        # Enable weights discriminator again
        self._d_model.trainable = True

        # Compile discriminator model
        self._d_model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(lr=0.0002, beta_1=0.5),
        )

        # Compute patch shape
        self._patch = self._d_model.output_shape[1]
        self._info('Patch shape ({0},{0}) ({1}/16)'.format(self._patch, self._image_shape[0]))
        self._register_session_data('patch', self._patch)

        # Add custom metrics, used by custom loss
        self._add_custom_metric('d_real_loss')  # Discriminator loss on real samples
        self._add_custom_metric('d_fake_loss')  # Discriminator loss on fake samples

        # Set stateful metrics
        self._custom_stateful_metrics = []

        # As this model does not converge, this will enable checkpoint
        self.plot = Pix2PixFloorPhotoModelPlot(self)

    def get_patch_size(self) -> int:
        """
        :return: Model patch size
        """
        return self._patch

    @staticmethod
    def unscale_image_range(
            image: 'np.ndarray',
            to_range: Tuple[Union[int, float], Union[int, float]]
    ) -> 'np.ndarray':
        """
        Scale back to normal image range.

        :param image: Scaled image
        :param to_range: Scale range
        :return: Unscaled image
        """
        return scale_array_to_range(
            array=image,
            to=to_range,
            dtype=None
        )

    def enable_early_stopping(self, *args, **kwargs) -> None:
        """
        See upper doc.
        """
        raise RuntimeError('Callback not available on current Model')

    def enable_reduce_lr_on_plateau(self, *args, **kwargs) -> None:
        """
        See upper doc.
        """
        raise RuntimeError('Callback not available on current Model')

    def _define_discriminator(self, input_shape: Tuple, output_shape: Tuple, df: int) -> 'Model':
        """
        Define discriminator model.

        :param input_shape: Input image shape (from input)
        :param output_shape: Generator output image shape
        :param df: Discriminator filters
        :return: Keras model
        """
        # Weight initialization
        init = RandomNormal(stddev=0.02)

        # Source image input
        in_src_image = Input(shape=input_shape)  # Input image

        # Target image input
        in_target_image = Input(shape=output_shape)  # Input from generator

        # Concatenate images channel-wise
        merged = Concatenate()([in_src_image, in_target_image])

        # C64
        d = Conv2D(df, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
        d = LeakyReLU(alpha=0.2)(d)

        # C128
        d = Conv2D(2 * df, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        # C256
        d = Conv2D(4 * df, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        # C512
        d = Conv2D(8 * df, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        # Second last output layer
        d = Conv2D(8 * df, (4, 4), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        # Patch output
        d = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
        patch_out = Activation('sigmoid', name='out_' + self._output_layers[0])(d)

        # Define model
        model = Model(inputs=[in_src_image, in_target_image], outputs=patch_out, name=self._output_layers[0])
        return model

    def _define_generator(self, input_shape: Tuple, output_shape: Tuple, gf: int) -> 'Model':
        """
        Define the standalone generator model.

        :param input_shape: Image input shape
        :param output_shape: Image output shape
        :param gf: Number of filters
        :return: Keras model
        """

        def define_encoder_block(
                layer_in: 'Layer',
                n_filters: int,
                batchnorm: bool = True
        ) -> 'Layer':
            """
            Encoder block.

            :param layer_in: Input layer
            :param n_filters: Number of filters
            :param batchnorm: Use batch normalization
            :return: Layer
            """
            # weight initialization
            _init = RandomNormal(stddev=0.02)
            # add downsampling layer
            _g = Conv2D(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=_init)(layer_in)
            # conditionally add batch normalization
            if batchnorm:
                _g = BatchNormalization()(_g, training=True)
            # leaky relu activation
            _g = LeakyReLU(alpha=0.2)(_g)
            return _g

        def decoder_block(
                layer_in: 'Layer',
                skip_in: 'Layer',
                n_filters: int,
                dropout: bool = True
        ):
            """
            Define decoder block.

            :param layer_in: Input layer
            :param skip_in: Skip layer
            :param n_filters: Number of filters
            :param dropout: Use dropout
            :return: Layer
            """
            # weight initialization
            _init = RandomNormal(stddev=0.02)
            # add upsampling layer
            _g = Conv2DTranspose(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=_init)(layer_in)
            # add batch normalization
            _g = BatchNormalization()(_g, training=True)
            # conditionally add dropout
            if dropout:
                _g = Dropout(0.5)(_g, training=True)
            # merge with skip connection
            _g = Concatenate()([_g, skip_in])
            # relu activation
            _g = Activation('relu')(_g)
            return _g

        # Weight initialization
        init = RandomNormal(stddev=0.02)

        # Image input
        in_image = Input(shape=input_shape)

        # Encoder model
        e1 = define_encoder_block(in_image, gf, batchnorm=False)
        e2 = define_encoder_block(e1, 2 * gf)
        e3 = define_encoder_block(e2, 4 * gf)
        e4 = define_encoder_block(e3, 8 * gf)
        e5 = define_encoder_block(e4, 8 * gf)
        if self._image_shape[0] >= 128:
            e6 = define_encoder_block(e5, 8 * gf)
        else:
            e6 = e5
        if self._image_shape[0] >= 256:
            e7 = define_encoder_block(e6, 8 * gf)
        else:
            e7 = e6

        # Bottleneck, no batch norm and relu
        b = Conv2D(8 * gf, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e7)
        b = Activation('relu')(b)

        # Decoder model
        if self._image_shape[0] >= 256:
            d1 = decoder_block(b, e7, 8 * gf)
        else:
            d1 = b
        if self._image_shape[0] >= 128:
            d2 = decoder_block(d1, e6, 8 * gf)
        else:
            d2 = b
        d3 = decoder_block(d2, e5, 512)
        d4 = decoder_block(d3, e4, 8 * gf, dropout=False)
        d5 = decoder_block(d4, e3, 4 * gf, dropout=False)
        d6 = decoder_block(d5, e2, 2 * gf, dropout=False)
        d7 = decoder_block(d6, e1, gf, dropout=False)

        # Output
        g = Conv2DTranspose(output_shape[2], (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d7)
        out_image = Activation('sigmoid', name='out_' + self._output_layers[1])(g)

        # Define model
        model = Model(inputs=in_image, outputs=out_image, name=self._output_layers[1])
        return model

    def generate_true_labels(self, n_samples: int) -> 'np.ndarray':
        """
        Returns true label vectors.

        :param n_samples: Number of samples
        :return: Vector
        """
        assert n_samples > 0
        return np.ones((n_samples, self._patch, self._patch, 1))

    def generate_fake_labels(self, n_samples: int) -> 'np.ndarray':
        """
        Returns fake label vectors.

        :param n_samples: Number of samples
        :return: Vector
        """
        assert n_samples > 0
        return np.zeros((n_samples, self._patch, self._patch, 1))

    def _generate_fake_samples(self, samples: 'np.ndarray') -> Tuple['np.ndarray', 'np.ndarray']:
        """
        Generate fake samples.

        :param samples: Image samples
        :return: Fake samples
        """
        # Generate fake instance
        x = self._g_model.predict(samples)

        # Create 'fake' class labels (0)
        y = self.generate_fake_labels(len(x))
        return x, y

    def _custom_train_function(self, inputs) -> List[float]:
        """
        Custom train function.
        inputs: (ximg + ylabel + yimg + ylabel_weights + yimg_weights + uses_learning_phase flag)

        :param inputs: Train input
        :return: Train metrics
        """
        assert len(inputs) == 6

        ximg_real: 'np.ndarray' = inputs[0]

        ylabel_real: 'np.ndarray' = inputs[1]
        yimg_real: 'np.ndarray' = inputs[2]

        ylabel_weights: 'np.ndarray' = inputs[3]
        yimg_weights: 'np.ndarray' = inputs[4]
        # use_learning_phase: int = inputs[5]

        # Generate a batch of fake samples
        yimg_fake, ylabel_fake = self._generate_fake_samples(yimg_real)

        self._d_model.trainable = True

        # Update discriminator for real samples
        d_real_loss = self._d_model.train_on_batch(
            x=[ximg_real, yimg_real],
            y=ylabel_real,
            sample_weight=ylabel_weights
        )

        # Update discriminator for generated samples
        d_fake_loss = self._d_model.train_on_batch(
            x=[ximg_real, yimg_fake],
            y=ylabel_fake,
            sample_weight=ylabel_weights
        )

        self._d_model.trainable = False

        # Update the generator, this does not train discriminator as weights
        # were defined as not trainable
        # 'loss', 'discriminator_loss', 'generator_loss', 'generator_iou', 'generator_accuracy'
        g_loss, gd_loss, gg_loss, g_iou, g_acc = self._model.train_on_batch(
            x=ximg_real,
            y=[ylabel_real, yimg_real],
            sample_weight=[ylabel_weights, yimg_weights]
        )

        del yimg_fake, ylabel_fake
        return [g_loss, gd_loss, gg_loss, g_iou, g_acc, d_real_loss, d_fake_loss]

    def _custom_val_function(self, inputs) -> List[float]:
        """
        Custom validation function.
        inputs: (ximg + ylabel + yimg + ylabel_weights + yimg_weights + uses_learning_phase flag)

        :param inputs: Train input
        :return: Validation metrics
        """
        assert len(inputs) == 6

        ximg_real: 'np.ndarray' = inputs[0]

        ylabel_real: 'np.ndarray' = inputs[1]
        yimg_real: 'np.ndarray' = inputs[2]

        ylabel_weights: 'np.ndarray' = inputs[3]
        yimg_weights: 'np.ndarray' = inputs[4]
        # use_learning_phase: int = inputs[5]

        # Generate a batch of fake samples
        yimg_fake, ylabel_fake = self._generate_fake_samples(yimg_real)

        # Evaluate discriminator for real samples
        d_real_loss = self._d_model.evaluate(
            x=[ximg_real, yimg_real],
            y=ylabel_real,
            sample_weight=ylabel_weights,
            verbose=False
        )

        # Evaluate discriminator for generated samples
        d_fake_loss = self._d_model.evaluate(
            x=[ximg_real, yimg_fake],
            y=ylabel_fake,
            sample_weight=ylabel_weights,
            verbose=False
        )

        # Evaluate the generator
        # 'loss', 'discriminator_loss', 'generator_loss', 'generator_iou', 'generator_accuracy'
        g_loss, gd_loss, gg_loss, g_iou, g_acc = self._model.evaluate(
            x=ximg_real,
            y=[ylabel_real, yimg_real],  # discriminator, generator
            sample_weight=[ylabel_weights, yimg_weights],
            verbose=False
        )

        del yimg_fake, ylabel_fake
        return [g_loss, gd_loss, gg_loss, g_iou, g_acc, d_real_loss, d_fake_loss]

    def _custom_epoch_finish_function(self, num_epoch: int) -> None:
        """
        Function triggered once each epoch finished.

        :param num_epoch: Number of the epoch
        """
        # Create figure
        _ = plt.figure(dpi=DEFAULT_PLOT_DPI)
        plt.style.use(DEFAULT_PLOT_STYLE)
        sample = self._samples[self._current_train_part]
        n_samples = len(sample['input'])
        plt.title(f'Epoch {num_epoch}')
        sample['predicted'] = self.predict_image(sample['input'])

        # Plot real source images
        for i in range(n_samples):
            plt.subplot(3, n_samples, 1 + i)
            plt.axis('off')
            plt.imshow(sample['input'][i])
        # Plot generated target image
        for i in range(n_samples):
            plt.subplot(3, n_samples, 1 + n_samples + i)
            plt.axis('off')
            plt.imshow(sample['predicted'][i])
        # Plot real target image
        for i in range(n_samples):
            plt.subplot(3, n_samples, 1 + n_samples * 2 + i)
            plt.axis('off')
            plt.imshow(sample['real'][i])

        fig_file: str = '{6}{0}{1}{2}_{3}_part_{4}_epoch{5}.png'.format(
            _PATH_LOGS, os.path.sep, self.get_name(True), self._current_train_date,
            self._current_train_part, num_epoch, self._path)
        plt.savefig(fig_file)
        plt.close()

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
        verbose = self._verbose

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
        free()

        # Get total parts to be processed
        n_parts: int = kwargs.get('n_parts', -1)
        assert isinstance(n_parts, int)
        assert total_parts - init_part >= n_parts >= 1 or n_parts == -1  # -1: no limits
        if n_parts != -1:
            print(f'Number of parts to be processed: {n_parts}')

        npt = 0  # Number of processed parts
        for i in range(total_parts):
            if i > 0:
                self._verbose = False
            part = i + 1
            if part < init_part:
                continue

            print(f'Loading data part {part}/{total_parts}')
            part_data = self._data.load_part(part=part, shuffle=True)
            xtrain_img: 'np.ndarray' = part_data['photo']
            ytrain_img: 'np.ndarray' = part_data['binary']
            ytrain_label = self.generate_true_labels(len(ytrain_img))
            free()

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
                ytrain=(ytrain_label, ytrain_img),
                xtest=None,
                ytest=None,
                epochs=epochs,
                batch_size=batch_size,
                val_split=val_split,
                shuffle=shuffle,
                use_custom_fit=True,
                continue_train=self._is_trained,
                compute_metrics=False
            )

            del xtrain_img, ytrain_img, ytrain_label, part_data
            free()
            if not self._is_trained:
                print('Train failed, stopping')
                self._verbose = verbose
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

        # Restore verbose
        self._verbose = verbose

    def predict(self, x: 'np.ndarray') -> 'np.ndarray':  # Image
        """
        See upper doc.
        """
        return self._model_predict(x=self._format_tuple(scale_array_to_range(x, (0, 1), 'uint8'), 'np', 'x'))[1]

    def evaluate(self, x: 'np.ndarray', y: Tuple['np.ndarray', 'np.ndarray']) -> List[float]:
        """
        See upper doc.
        """
        return self._model_evaluate(
            x=self._format_tuple(x, 'np', 'x'),
            y=self._format_tuple(y, ('np', 'np'), 'y')
        )
