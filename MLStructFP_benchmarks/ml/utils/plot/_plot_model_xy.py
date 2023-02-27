"""
MLSTRUCTFP BENCHMARKS - ML - MODEL - UTILS - PLOT - MODEL XY PLOT

Plots model xy.
"""

__all__ = ['GenericModelPlotXY']

from MLStructFP_benchmarks.ml.utils.plot._plot_model import GenericModelPlot
from MLStructFP_benchmarks.ml.utils import r2_score
from MLStructFP.utils import save_figure, configure_figure, DEFAULT_PLOT_DPI, \
    DEFAULT_PLOT_FIGSIZE

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from typing import TYPE_CHECKING, Tuple, Optional
import math
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from MLStructFP_benchmarks.ml.model.core import GenericModelXY


class GenericModelPlotXY(GenericModelPlot):
    """
    Model XY plot.
    """
    _model: 'GenericModelXY'

    def __init__(self, model: 'GenericModelXY') -> None:
        """
        Constructor.

        :param model: Model object
        """
        GenericModelPlot.__init__(self, model)

    def _confusion_matrix(
            self,
            x: 'np.ndarray',
            y: 'np.ndarray',
            x_label: str,
            y_label: str,
            column: str,
            bins: int,
            percentage: bool = False,
            percentage_mode: str = 'y',  # 'x': sum 100% by predicted, if 'y': sum 100% by true
            normalize: bool = False,
            min_lim: Optional[float] = None,
            max_lim: Optional[float] = None,
            round_f: int = 3,
            fix_ylims: bool = False,
            colormap: str = 'Blues',
            val_lims: Tuple[Optional[float], Optional[float]] = (None, None),
            save: str = '',
            **kwargs
    ) -> None:
        """
        Generate confusion matrix from col.

        :param x: X vector
        :param y: Y vector
        :param x_label: Label on X axis
        :param y_label: Label on Y axis
        :param column: Column name
        :param bins: Number of bins
        :param percentage: Use percentages instead of total occurrences
        :param percentage_mode: Percentage mode, x or y
        :param normalize: Normalize data
        :param min_lim: Min limit
        :param max_lim: Max limit
        :param round_f: Decimals to round
        :param fix_ylims: Fix ylims on certain matplotlib versions
        :param colormap: Name of the colormap to be used
        :param val_lims: Value limits (min, max), if None no limit is applied
        :param save: Save figure to file
        :param kwargs: Optional keyword arguments
        """
        assert bins > 1, 'Invalid number of bins'
        assert percentage_mode in ['x', 'y'], 'Invalid percentage mode'
        col_y, _ = self._model.get_column_index(column=column, xy='y')

        val_lim_min: float = val_lims[0]
        if val_lim_min is None:
            val_lim_min = -math.inf
        val_lim_max: float = val_lims[1]
        if val_lim_max is None:
            val_lim_max = math.inf
        assert val_lim_min < val_lim_max, 'Value limits are switched'

        # Create ranges from minimum and maximum values
        computed_min = max(min(min(x), min(y)), val_lim_min)
        computed_max = min(max(max(x), max(y)), val_lim_max)
        if min_lim is None:
            min_lim = computed_min
        else:
            min_lim = max(min_lim, computed_min)
        if max_lim is None:
            max_lim = computed_max
        else:
            max_lim = min(max_lim, computed_max)
        assert max_lim > min_lim, \
            f'Max limit <{max_lim}> should be at least <{min_lim}> and lower than <{computed_max}>'

        # Create steps range
        dd = (max_lim - min_lim) / bins

        # Create classes
        classes = []  # String list
        classes_range = []
        charsplit = ' ─ '
        for j in range(bins):
            classes_range.append([min_lim + dd * j, min_lim + dd * (j + 1)])
            classes.append('{0}{2}{1}'.format(
                round(min_lim + dd * j, round_f), round(min_lim + dd * (j + 1), round_f), charsplit)
            )
        classes.append(kwargs.get('other_label', 'Other'))
        clsx = {}
        for k in range(bins + 1):
            clsx[k] = 0  # Last is other

        def _find(_dat: float) -> int:
            """
            Find class of data.
            """
            for _j in range(bins):
                if classes_range[_j][0] <= _dat <= classes_range[_j][1]:
                    return _j
            return bins  # 'Other'

        x_class = []
        y_class = []

        total_other = 0
        # Create class range
        for j in range(len(x)):
            cx = _find(max(min(x[j], val_lim_max), val_lim_min))
            x_class.append(cx)
            clsx[cx] += 1
            cy = _find(max(min(y[j], val_lim_max), val_lim_min))
            y_class.append(cy)
            if cx == bins:
                total_other += 1
            if cy == bins:
                total_other += 1

        if total_other == 0:
            classes.pop()  # Remove 'Other'

        # Remove null classes in x
        replace_class = {}
        if kwargs.get('replace_class', True):
            for k in clsx.keys():
                if clsx[k] == 0:
                    if k == 0:  # First class should be fixed by user modifying val_lim_min param
                        continue
                    to_cls = 0
                    for j in range(k - 1):  # Find the lower class that contains elements
                        if clsx[k - j - 1] != 0:
                            to_cls = k - j - 1
                            break
                    if k >= len(classes) - 1:
                        continue
                    replace_class[k] = to_cls
                    classes[to_cls] = classes[to_cls].split(charsplit)[0] + charsplit + classes[k].split(charsplit)[1]

        # Replace classes in y:
        replk = list(replace_class.keys())

        # Remove classes
        new_yclass = []
        for n in range(len(y_class)):
            if y_class[n] not in replk:
                new_yclass.append(y_class[n])
            else:
                new_yclass.append(replace_class[y_class[n]])
        y_class = new_yclass
        new_classes = []
        for n in range(len(classes)):
            if n not in replk:
                new_classes.append(classes[n])
        classes = new_classes

        # Create confusion matrix
        cm = confusion_matrix(y_class, x_class)  # true, predict
        if kwargs.get('print_accuracy', False):
            print('Accuracy:', accuracy_score(y_class, x_class))
        if kwargs.get('print_classification_report', False):
            print(classification_report(y_class, x_class, target_names=classes))

        # Percentage mode
        if percentage:
            cm2 = np.zeros((len(cm), len(cm)))
            for i in range(len(cm)):
                s = 0
                if percentage_mode == 'x':
                    for j in range(len(cm)):
                        s += cm[i][j]
                elif percentage_mode == 'y':
                    for j in range(len(cm)):
                        s += cm[j][i]
                s = float(s)
                if s == 0:
                    continue
                for j in range(len(cm)):
                    if percentage_mode == 'x':
                        cm2[i, j] = float(1.0 * cm[i][j]) / s * 100
                    elif percentage_mode == 'y':
                        cm2[j, i] = float(1.0 * cm[j][i]) / s * 100
            cm = cm2

        # Compute R2
        r2 = r2_score(x, y, False)[0]

        # Create figure
        title = '{2} - {0}\nR²={1}'.format(
            self._model.get_column_name('y', col_y).capitalize(), r2, kwargs.get('title', 'Confusion matrix')
        )

        fig, ax = plt.subplots(figsize=(DEFAULT_PLOT_FIGSIZE, DEFAULT_PLOT_FIGSIZE), dpi=DEFAULT_PLOT_DPI)
        im = ax.imshow(cm, interpolation='nearest', cmap=colormap)
        # ax.grid()
        ax.figure.colorbar(im, ax=ax, shrink=0.725, pad=0.03, aspect=25)
        # We want to show all ticks*...
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes, yticklabels=classes
        )
        plt.ylabel(y_label + kwargs.get('ylabel_units', ''))
        plt.xlabel(x_label + kwargs.get('xlabel_units', ''))
        if title is not None and kwargs.get('show_title', True):
            ax.set_title(f'{title}', fontsize=kwargs.get('cfg_fontsize_title', 20))

        # Rotate axis
        plt.setp(
            ax.get_xticklabels(),
            rotation=45,
            ha='right',
            rotation_mode='anchor'
        )

        # Write total data
        fmt = '2f' if normalize else 'd'
        if percentage:
            fmt = '.1f'

        # noinspection PyArgumentList
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                d = cm[i, j]
                if percentage:
                    if d < 0.1:
                        d = 0
                if d != 0:
                    d = format(d, fmt)
                ax.text(j, i, d,
                        ha='center',
                        va='center',
                        color='white' if cm[i, j] > thresh else 'black',
                        fontsize=kwargs.get('cfg_fontsize_numbers', 9))

        if fix_ylims:
            ylims = ax.get_ylim()
            ax.set_ylim(math.ceil(ylims[0]) + 0.5, math.floor(ylims[1]) - 0.5)
        ax.invert_yaxis()
        kwargs['cfg_tight_layout'] = True
        kwargs['cfg_grid'] = False
        configure_figure(**kwargs)
        save_figure(save, **kwargs)
        plt.show()
