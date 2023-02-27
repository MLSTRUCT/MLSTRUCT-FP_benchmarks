"""
MLSTRUCTFP BENCHMARKS - ML - MODEL - UTILS - PLOT - ARCHITECTURES - PIX2PIX FLOOR PHOTO MODEL

Model plot.
"""

__all__ = ['Pix2PixFloorPhotoModelPlot']

from MLStructFP_benchmarks.ml.utils.plot._plot_model import GenericModelPlot
from MLStructFP.utils import save_figure, configure_figure, DEFAULT_PLOT_DPI, DEFAULT_PLOT_STYLE

from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from MLStructFP_benchmarks.ml.model.architectures import Pix2PixFloorPhotoModel


class Pix2PixFloorPhotoModelPlot(GenericModelPlot):
    """
    PIX2PIX Model plot.
    """
    _model: 'Pix2PixFloorPhotoModel'

    def __init__(self, model: 'Pix2PixFloorPhotoModel') -> None:
        """
        Constructor.

        :param model: Model object
        """
        super().__init__(model)

    def plot_predict(self, im: 'np.ndarray', save: str = '', **kwargs) -> None:
        """
        Predict and plot image.

        :param im: Image
        :param save: Save figure to file
        :param kwargs: Optional keyword arguments
        """
        assert len(im.shape) == 3
        im_pred = self._model.predict_image(im)

        kwargs['cfg_grid'] = False
        fig = plt.figure(dpi=DEFAULT_PLOT_DPI)
        # fig.subplots_adjust(hspace=0.5)
        plt.axis('off')

        ax1: 'plt.Axes' = fig.add_subplot(131)
        ax1.title.set_text('Input')
        ax1.imshow(im / 255)
        plt.xlabel('x $(px)$')
        plt.ylabel('y $(px)$')
        plt.axis('off')
        configure_figure(**kwargs)

        ax2 = fig.add_subplot(132)
        ax2.title.set_text('Output')
        ax2.imshow(im_pred / 255)
        # plt.xlabel('x $(px)$')
        plt.axis('off')
        configure_figure(**kwargs)
        save_figure(save, **kwargs)
        plt.show()

    def plot_sample(self, part: int, sample: int, domain: str = 'predicted', save: str = '', **kwargs) -> None:
        """
        Plot single predicted image sample.

        :param part: Part number
        :param sample: Sample number
        :param domain: Which data to plot
        :param save: Save figure to file
        :param kwargs: Optional keyword arguments
        """
        assert domain in ['predicted', 'real', 'input'], \
            'Invalid domain value, expected "predicted", "real" or "input"'

        # noinspection PyProtectedMember
        samples = self._model._samples
        domain_sample = samples[part][domain]
        assert part in list(samples.keys()), f'Part <{part}> does not exists on samples'
        assert 1 <= sample <= len(domain_sample)
        plt.figure(dpi=DEFAULT_PLOT_DPI)
        plt.axis('off')
        plt.imshow(domain_sample[sample - 1] / 255)
        kwargs['cfg_grid'] = False
        configure_figure(**kwargs)
        save_figure(save, **kwargs)
        plt.show()

    def summarize_performance(self, part: int, save: str = '', **kwargs) -> None:
        """
        Plot examples from part to see model generation performance.

        :param part: Part number
        :param save: Save figure to file
        :param kwargs: Optional keyword arguments
        """
        # noinspection PyProtectedMember
        samples = self._model._samples
        assert part in list(samples.keys()), f'Part <{part}> does not exists on samples'

        sample = samples[part]
        n_samples = len(sample['input'])
        if n_samples == 0:
            raise ValueError(f'Part <{part}> does not have any samples to show')

        # Create figure
        plt.figure(dpi=DEFAULT_PLOT_DPI)
        plt.style.use(DEFAULT_PLOT_STYLE)

        # plot real source images
        for i in range(n_samples):
            plt.subplot(3, n_samples, 1 + i)
            plt.axis('off')
            plt.imshow(sample['input'][i] / 255)
        # plot generated target image
        for i in range(n_samples):
            plt.subplot(3, n_samples, 1 + n_samples + i)
            plt.axis('off')
            plt.imshow(sample['predicted'][i] / 255)
        # plot real target image
        for i in range(n_samples):
            plt.subplot(3, n_samples, 1 + n_samples * 2 + i)
            plt.axis('off')
            plt.imshow(sample['real'][i] / 255)

        save_figure(save, **kwargs)
        plt.show()
