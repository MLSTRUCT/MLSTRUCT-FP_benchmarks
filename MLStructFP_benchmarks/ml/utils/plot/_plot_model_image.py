"""
MLSTRUCTFP BENCHMARKS - ML - MODEL - UTILS - PLOT - PLOT MODEL IMAGE

Plot image models.
"""

__all__ = ['GenericModelPlotImage']

from MLStructFP_benchmarks.ml.utils import binary_accuracy_metric
from MLStructFP_benchmarks.ml.utils.plot._plot_model import GenericModelPlot
from MLStructFP.utils import save_figure, configure_figure, DEFAULT_PLOT_DPI, DEFAULT_PLOT_FIGSIZE

from matplotlib.ticker import PercentFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import TYPE_CHECKING, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from MLStructFP_benchmarks.ml.model.core import GenericModelImage


class GenericModelPlotImage(GenericModelPlot):
    """
    Model Image plot.
    """
    _model: 'GenericModelImage'

    def __init__(self, model: 'GenericModelImage') -> None:
        """
        Constructor.

        :param model: Model object
        """
        GenericModelPlot.__init__(self, model)

    # noinspection PyMethodMayBeStatic
    def _plot_accuracy_df(self, df: 'pd.DataFrame', **kwargs) -> None:
        """
        Plot accuracy dataframe.

        :param df: Dataframe
        :param kwargs: Optional keyword arguments
        """
        plt.figure(figsize=(DEFAULT_PLOT_FIGSIZE, DEFAULT_PLOT_FIGSIZE), dpi=DEFAULT_PLOT_DPI)
        x = df['accuracy']
        print('Mean:', np.mean(x))
        if kwargs.get('use_percentage', True):
            plt.hist(x, bins=kwargs.get('bins', 100), weights=np.ones(len(x)) / len(x))
            plt.ylabel('Percentage')
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=kwargs.get('num_decimals', 1)))
        else:
            plt.hist(x, bins=kwargs.get('bins', 100))
            plt.ylabel('Frequency')
        plt.xlabel('Accuracy')
        configure_figure(**kwargs)
        save_figure(save=kwargs.get('save_figure', ''), **kwargs)

    def _plot_image_comparision(
            self,
            column_id: str,
            obj_id: int,
            xy: str,
            x_img: 'np.ndarray',
            y_img: 'np.ndarray',
            y_pred: 'np.ndarray',
            metrics: List[float],
            save: str = '',
            **kwargs
    ) -> None:
        """
        Compare predicted and real images from model.

        :param column_id: Column ID name
        :param obj_id: Object ID
        :param xy: Which dataframe
        :param x_img: Image from X
        :param y_img: Image from Y
        :param y_pred: Predicted image Y
        :param metrics: Computed metrics
        :param save: Save figure to file
        :param kwargs: Optional keyword arguments
        """
        # Find which value from acc is accuracy
        acc: float = -1
        j = 0
        for k in self._model.get_metric_names():
            if 'val_' in k:
                continue
            if 'accuracy' in k:
                acc = metrics[j]
            j += 1
        if acc == -1:
            raise RuntimeError('Accuracy not found at metrics, check out if accuracy on loss')

        # Configure title if not exists
        if 'cfg_fontsize_title' not in kwargs.keys():
            kwargs['cfg_fontsize_title'] = 26

        kwargs['cfg_grid'] = False
        fig = plt.figure(figsize=(2 * DEFAULT_PLOT_FIGSIZE, DEFAULT_PLOT_FIGSIZE), dpi=DEFAULT_PLOT_DPI)
        # fig.subplots_adjust(hspace=0)
        plt.axis('off')
        if kwargs.get('title', True):
            plt.title('{4}\nObject {0} {1} ─ {2}\nAccuracy: {3:.4f}'.format(
                column_id, obj_id, xy, acc, self._model.get_name()))
        rf = kwargs.get('roundf', 3)
        print(f'Acc: {round(acc, rf)} from source {round(binary_accuracy_metric(x_img, y_img), rf)}')
        configure_figure(**kwargs)

        cmapcolor = kwargs.get('cmap_img', 'jet')  # gray, binary, jet
        ax1: 'plt.Axes' = fig.add_subplot(131)
        ax1.title.set_text('Architecture')
        ax1.imshow(x_img, cmap=cmapcolor)
        plt.xlabel('x $(px)$')
        plt.ylabel('y $(px)$')
        plt.axis('off')
        configure_figure(**kwargs)

        ax2 = fig.add_subplot(132)
        ax2.title.set_text('Engineering')
        ax2.imshow(y_img, cmap=cmapcolor)
        # plt.xlabel('x $(px)$')
        plt.axis('off')
        configure_figure(**kwargs)

        ax3 = fig.add_subplot(133)
        ax3.title.set_text('Predicted')
        im = ax3.imshow(y_pred, cmap=cmapcolor)
        plt.axis('off')
        configure_figure(**kwargs)
        if kwargs.get('prob_cbar_enabled', False):
            divider = make_axes_locatable(ax3)
            cax = divider.append_axes('right', size='2.5%', pad=kwargs.get('prob_colorbar_pad', 0.15))
            cb = plt.colorbar(im, cax=cax)
            cb.set_label(kwargs.get('prob_colorbar_title', 'Probability'))
            # plt.xlabel('x $(px)$')
        plt.subplots_adjust(wspace=0.075, hspace=0)

        save_figure(save, **kwargs)
        plt.show()
