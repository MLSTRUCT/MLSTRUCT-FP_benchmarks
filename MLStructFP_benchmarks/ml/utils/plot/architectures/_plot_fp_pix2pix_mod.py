"""
MLSTRUCTFP BENCHMARKS - ML - MODEL - UTILS - PLOT - ARCHITECTURES - PIX2PIX FLOOR PHOTO MODEL MOD

Model plot.
"""

__all__ = ['Pix2PixFloorPhotoModelModPlot']

from MLStructFP_benchmarks.ml.utils.plot._plot_model import GenericModelPlot
from MLStructFP.utils import save_figure, configure_figure, DEFAULT_PLOT_DPI

from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
import os

if TYPE_CHECKING:
    from MLStructFP_benchmarks.ml.model.architectures import Pix2PixFloorPhotoModModel


class Pix2PixFloorPhotoModelModPlot(GenericModelPlot):
    """
    PIX2PIX Model plot.
    """
    _model: 'Pix2PixFloorPhotoModModel'

    def __init__(self, model: 'Pix2PixFloorPhotoModModel') -> None:
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
        im_pred = self._model.predict(im)

        kwargs['cfg_grid'] = False
        fig = plt.figure(dpi=DEFAULT_PLOT_DPI)
        # fig.subplots_adjust(hspace=.5)
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

    # noinspection PyProtectedMember
    def samples(self, epoch: int, batch_i: int, save: bool = False, savename: str = '', **kwargs) -> None:
        """
        Sample example images to folder.

        :param epoch: Epoch
        :param batch_i: Batch num
        :param save: Save file
        :param savename: Save figure to file
        :param kwargs: Optional keyword arguments
        """
        sample_dir: str = ''
        if save:
            sample_dir = '%s/%s_%s/images_%s_part%s' % (
                self._model._path_logs, self._model.get_name(True), self._model._train_date, self._model._dir,
                self._model._train_current_part
            )
            os.makedirs(sample_dir, exist_ok=True)
        r, c = 3, 3

        imgs_a, imgs_b = self._model._load_data_samples(n=3)  # Range (0, 255)
        fake_a = self._model.predict(imgs_b)

        gen_imgs = np.concatenate([imgs_b, fake_a, imgs_a])
        gen_imgs /= 255

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c, dpi=DEFAULT_PLOT_DPI)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i], fontsize=6, pad=5)
                axs[i, j].axis('off')
                cnt += 1

        configure_figure(**kwargs)
        save_figure(savename, **kwargs)
        if save:
            fig.savefig('%s/%d_%d.png' % (sample_dir, epoch, batch_i))
            plt.close()
        else:
            plt.show()
