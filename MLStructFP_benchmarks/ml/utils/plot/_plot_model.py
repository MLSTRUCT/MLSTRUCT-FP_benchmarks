"""
MLSTRUCTFP BENCHMARKS - ML - MODEL - UTILS - PLOT - PLOT MODEL

Model plotting methods.
"""

__all__ = [
    'compare_metrics_from_csv',
    'GenericModelPlot'
]

from MLStructFP_benchmarks.ml.utils import load_history_from_csv
from MLStructFP_benchmarks.ml.utils.plot import plot_model_architecture
from MLStructFP.utils import save_figure, configure_figure, DEFAULT_PLOT_DPI, DEFAULT_PLOT_FIGSIZE

from IPython import display
from matplotlib.ticker import MaxNLocator
from typing import TYPE_CHECKING, Union, Tuple, List, Dict, Any, Optional
import math
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from MLStructFP_benchmarks.ml.model.core import GenericModel
    from matplotlib.figure import Figure

_PATH_SESSION = '.session'


def _assert_metric_in_history(history: Dict[str, Any], metric: str, model_name: str) -> None:
    """
    Check metric exists on history dict.

    :param history: History metrics train dict
    :param metric: Metric to check
    :param model_name: Model name of the history
    """
    if metric not in history.keys():
        valk: List[str] = []
        for k in history.keys():
            if 'val_' in k:
                continue
            valk.append(k)
        raise ValueError('Metric <{0}> does not exist in model <{1}>. {2}'.format(
            metric, model_name, 'Available metrics: <{0}>'.format(', '.join(valk))))


def compare_metrics_from_csv(
        csv_file: Union[str, List[str]],
        csv_metric: Union[str, List[str]],
        csv_model_name: Union[str, List[str]],
        csv_factor: Optional[Union[Union[int, float], List[Union[int, float]]]] = None,
        lw: Union[int, float] = 2,
        val: bool = False,
        val_only: bool = False,
        limx: int = 0,
        title: str = '',
        leg_size: int = 10,
        plot_style: Union[str, List[str]] = '.-',
        plot_val_style: Union[str, List[str]] = '.-',
        save: str = '',
        yscale: str = 'linear',
        ylabel: str = 'Metric value',
        square: bool = True,
        **kwargs
) -> Tuple['Figure', 'plt.Axes', int]:
    """
    Compare history metrics.

    :param csv_file: File to load the metrics
    :param csv_metric: Metric to plot from csv
    :param csv_model_name: Model name from csv
    :param csv_factor: Factor to multiply each plot, if None 1 will be applied
    :param lw: Linewidth
    :param val: Plot validation
    :param val_only: Plot only validation
    :param limx: Limit at epoch
    :param title: Figure title
    :param leg_size: Legend size
    :param plot_style: Train line style
    :param plot_val_style: Validation line style
    :param save: Save figure
    :param yscale: Yscale
    :param ylabel: Ylabel
    :param square: Use squared plots
    :param kwargs: Optional keyword arguments
    :return: Figure, maxepoch
    """
    assert leg_size > 0, 'Invalid legend size'
    assert lw > 0, 'Invalid linewidth'

    if not isinstance(csv_file, list):
        csv_file = [csv_file]
    if not isinstance(csv_metric, list):
        csv_metric = [csv_metric]
    if not isinstance(csv_model_name, list):
        csv_model_name = [csv_model_name]
    if csv_factor is None:
        csv_factor = []
        for k in range(len(csv_file)):
            csv_factor.append(1.0)
    assert isinstance(csv_factor, list)

    if val_only:
        val = True

    # Create figure
    if square:
        fig = plt.figure(figsize=(DEFAULT_PLOT_FIGSIZE, DEFAULT_PLOT_FIGSIZE), dpi=DEFAULT_PLOT_DPI)
    else:
        fig = plt.figure(dpi=DEFAULT_PLOT_DPI)
    ax = plt.axes()

    # Iterate through each csv
    assert len(csv_file) == len(csv_model_name), 'CSV list must have the same length'
    assert len(csv_file) == len(csv_metric), 'CSV list must have the same length'

    if not isinstance(plot_style, list):
        plot_style = [plot_style] * len(csv_file)
    if not isinstance(plot_val_style, list):
        plot_val_style = [plot_val_style] * len(csv_file)

    maxepoch: int = 0
    for i in range(len(csv_file)):
        assert isinstance(csv_file[i], str)
        assert isinstance(csv_metric[i], str)
        assert isinstance(csv_model_name[i], str)

        csv_history_i = load_history_from_csv(csv_file[i])
        csv_metric_i = csv_metric[i].lower()
        if 'val_' in csv_metric_i:
            csv_metric_i = csv_metric_i.replace('val_', '', 1)
            print(f'Using train csv metric instead of validation at {i}')

        # Assert metrics
        _assert_metric_in_history(csv_history_i, csv_metric_i, csv_model_name[i])

        # Compute max epochs list
        maxepoch_csv = len(csv_history_i[list(csv_history_i.keys())[0]])
        x_csv: List[int] = [*range(1, maxepoch_csv + 1)]
        maxepoch = max(maxepoch, maxepoch_csv)

        # Plot csv model
        _n: str
        if kwargs.get('legend_bold', False):
            csv_model_name_i = '$\\bf{' + csv_model_name[i].replace(' ', '\ ') + '}$'
        else:
            csv_model_name_i = csv_model_name[i]
        if not val_only:
            if kwargs.get('add_metric_to_legend', True):
                _n = f'{csv_model_name_i} {csv_metric_i}'
            else:
                _n = csv_model_name_i
            if kwargs.get('use_metric_simple_names', False):
                _n = 'Training'
            ax.plot(x_csv, np.array(csv_history_i[csv_metric_i]) * csv_factor[i], plot_style[i],
                    linewidth=lw, label=_n)
        if 'val_' + csv_metric_i in csv_history_i.keys() and val:
            if kwargs.get('add_metric_to_legend', True):
                _n = f'{csv_model_name_i} val_{csv_metric_i}'
            else:
                _n = csv_model_name_i
            if kwargs.get('use_metric_simple_names', False):
                _n = 'Validation'
            ax.plot(x_csv, np.array(csv_history_i['val_' + csv_metric_i]) * csv_factor[i], plot_val_style[i],
                    linewidth=lw, label=_n)
        del csv_metric_i

    ax.legend(loc=kwargs.get('legend_loc', 'best'), prop={'size': leg_size})
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.grid(b=True, which='major')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # X limit
    limx = min(limx, maxepoch)
    # if limx > maxepoch:
    #     raise ValueError('Epoch limit cannot exceed max epochs')
    if limx <= 1:
        limx = maxepoch
    if limx > 1:
        plt.xlim([1, limx])

    # Set legend
    if title != '':
        plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Epoch')
    plt.yscale(yscale)
    kwargs['cfg_tight_layout'] = True
    configure_figure(**kwargs)
    save_figure(save, **kwargs)
    plt.show()
    return fig, ax, maxepoch


def _data_mark_on_plot(
        x: Union['np.ndarray', List[Union[int, float]]],
        y: Union['np.ndarray', List[Union[int, float]]],
        mode: str, plot: 'plt.Line2D',
        format_name: str,
        ylims: List[Union[int, float]]
) -> None:
    """
    Mark data on plot.

    :param x: X
    :param y: Y
    :param mode: Mark mode
    :param plot: Last line plot
    :param format_name: Name string
    :param ylims: Limits of the plot
    """
    mode = str(mode).lower()
    c: str = plot[0].get_color()  # Line color
    lw: float = 0.5 * plot[0].get_linewidth()
    if mode == 'none':
        return
    elif mode == 'max':
        _max = -math.inf
        _maxp = 0
        for j in range(len(y)):
            if y[j] > _max:
                _max = y[j]
                _maxp = j
        plt.axvline(x[_maxp], color=c, linestyle='dashed', linewidth=lw, label=None)
        text = 'max ' + format_name.format(_max, _maxp)
        plt.text(x[_maxp], ylims[0], text, fontsize=10,
                 rotation=90, rotation_mode='anchor', color=c, horizontalalignment='left')
        plt.plot(x[_maxp], _max, '.', color=c, markersize=10 * lw)
    elif mode == 'min':
        _min = math.inf
        _minp = 0
        for j in range(len(y)):
            if y[j] < _min:
                _min = y[j]
                _minp = j
        plt.axvline(x[_minp], color=c, linestyle='dashed', linewidth=lw, label=None)
        text = 'min ' + format_name.format(_min, _minp)
        plt.text(x[_minp], ylims[1], text, fontsize=10,
                 rotation=90, rotation_mode='anchor', color=c, horizontalalignment='right')
        plt.plot(x[_minp], _min, '.', color=c, markersize=10 * lw)
    else:
        raise ValueError(f'Invalid mark mode <{mode}>')


class GenericModelPlot(object):
    """
    Model plot.
    """
    _model: 'GenericModel'

    def __init__(self, model: Union['GenericModel', Any]) -> None:
        """
        Constructor.

        :param model: Model object
        """
        self._model = model

    def architecture(
            self,
            vertical: bool = True,
            dpi: int = 200,
            save: bool = True,
            version: int = 1
    ) -> Optional['display.Image']:
        """
        Visualize model architecture as a plot.
        https://datascience.stackexchange.com/questions/12851/how-do-you-visualize-neural-network-architectures/44571

        :param vertical: Vertical plot
        :param dpi: Image output resolution
        :param save: Save file
        :param version: Which version to use
        """
        assert dpi > 0, 'Invalid DPI format'
        if save:
            out_file = f"{_PATH_SESSION}\\{self._model.get_name(True) + '_plot.png'}"
        else:
            out_file = ''
        rankdir = 'TB'
        if not vertical:
            rankdir = 'LB'

        # noinspection PyProtectedMember
        return plot_model_architecture(
            self._model._model,
            show_shapes=True,
            expand_nested=True,
            rankdir=rankdir,
            dpi=dpi,
            to_file=out_file,
            version=version
        )

    def train_metrics(
            self,
            metric: Union[str, List[str]],
            lw: Union[int, float] = 2,
            val: bool = True,
            data_mark: str = 'none',
            save: str = '',
            **kwargs
    ) -> None:
        """
        Plot suite results.

        :param metric: Metric(s) to plot
        :param lw: Linewidth
        :param val: Plot validation
        :param data_mark: Use marks on data
        :param save: Save figure
        :param kwargs: Optional keyword arguments
        """
        assert lw > 0, 'Invalid linewidth'

        # Get model history
        history = self._model.get_train_history()

        if not isinstance(metric, list):
            metric = [metric]
        for k in range(len(metric)):
            metric[k] = metric[k].lower()
            assert isinstance(metric[k], str), f'Metric at pos <{k}> expected to be string'
            if 'val_' in metric[k]:
                metric[k] = metric[k].replace('val_', '', 1)
                print(f'Using train metric instead of validation at pos <{k}>')
            _assert_metric_in_history(history, metric[k], self._model.get_name())

        # Plot metrics
        maxepoch = self._model.get_total_epochs()
        x: List[int] = [*range(1, maxepoch + 1)]

        # Create figure
        plt.figure(dpi=DEFAULT_PLOT_DPI)
        ax = plt.axes()

        # Compute metrics
        ylims: List[float] = [math.inf, -math.inf]
        for m in metric:
            if 'val_' + m in history.keys() and val:
                ylims[0] = min(ylims[0], min(history['val_' + m]))
                ylims[1] = max(ylims[1], max(history['val_' + m]))
            ylims[0] = min(ylims[0], min(history[m]))
            ylims[1] = max(ylims[1], max(history[m]))

        # Plot metrics
        for m in metric:
            m_leg: str = m.replace('_', '\_')
            if 'val_' + m in history.keys() and val:
                p = ax.plot(x, history['val_' + m], linewidth=lw, label=r'$\bf{' + m_leg + r'}$ Validation')
                _data_mark_on_plot(x, history['val_' + m], data_mark, p, '{0:.3f} at epoch {1}', ylims)
            p = ax.plot(x, history[m], linewidth=lw, label=r'$\bf{' + m_leg + r'}$ Training')
            _data_mark_on_plot(x, history[m], data_mark, p, '{0:.3f} at epoch {1}', ylims)
        ax.legend(loc='best')

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.grid(b=True, which='major')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlim([1, maxepoch])

        # Set legend
        if len(metric) == 1:
            plt.title(r'$\bf{Train\ ' + metric[0].replace('_', '\_') + '}$')
            plt.ylabel(metric[0])
        else:
            plt.title(r'$\bf{Train\ metrics}$')
            plt.ylabel('Values')
        plt.xlabel('Epoch')
        configure_figure(**kwargs)
        save_figure(save, **kwargs)
        plt.show()

    def compare_metrics_from_csv(
            self,
            csv_file: Union[str, List[str]],
            csv_metric: Union[str, List[str]],
            csv_model_name: Union[str, List[str]],
            metric: Union[str, List[str]],
            fig_size: int = 6,
            lw: Union[int, float] = 2,
            val: bool = False,
            limx: int = 0,
            leg_size: int = 10,
            save: str = '',
            **kwargs
    ) -> None:
        """
        Compare history metrics.

        :param csv_file: File to load the metrics
        :param csv_metric: Metric to plot from csv
        :param csv_model_name: Model name from csv
        :param metric: Metric to plot from model
        :param fig_size: Figure size
        :param lw: Linewidth
        :param val: Plot validation
        :param limx: Limit at epoch
        :param leg_size: Legend size
        :param save: Save figure
        :param kwargs: Optional keyword arguments
        """
        fig, ax, maxepochs = compare_metrics_from_csv(
            csv_file=csv_file,
            csv_metric=csv_metric,
            csv_model_name=csv_model_name,
            fig_size=fig_size,
            lw=lw,
            val=val,
            leg_size=leg_size
        )
        # noinspection PyUnresolvedReferences
        plt.figure(fig.number)

        # Plot model
        history = self._model.get_train_history()
        metric = metric.lower()

        if 'val_' in metric:
            metric = metric.replace('val_', '', 1)
            print('Using train metric instead of validation')

        # Plot model
        _assert_metric_in_history(history, metric, self._model.get_name())
        maxepoch_model = self._model.get_total_epochs()
        x_model: List[int] = [*range(1, maxepoch_model + 1)]
        model_name = '$\\bf{' + self._model.get_name().replace(' ', '\ ') + '}$'
        if 'val_' + metric in history.keys() and val:
            _n: str = f'{model_name} val_{metric}'
            ax.plot(x_model, history['val_' + metric], linewidth=lw, label=_n)
        _n: str = f'{model_name} {metric}'
        ax.plot(x_model, history[metric], linewidth=lw, label=_n)
        maxepochs = max(maxepochs, maxepoch_model)
        ax.legend(loc='best', prop={'size': leg_size})

        # X limit
        if limx > maxepochs:
            raise ValueError('Epoch limit cannot exceed max epochs')
        if limx <= 0:
            limx = maxepochs
        plt.xlim([1, limx])

        # Set legend
        plot_title = r'$\bf{Train\ ' + model_name.replace('$', '') + '\ comparision}$'
        plt.ylabel(f'Metric {metric} value')
        plt.title(plot_title)
        configure_figure(**kwargs)
        save_figure(save, **kwargs)
        plt.show()
