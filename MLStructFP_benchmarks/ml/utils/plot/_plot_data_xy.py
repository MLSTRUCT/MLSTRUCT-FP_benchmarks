"""
MLSTRUCTFP BENCHMARKS - ML - MODEL - UTILS - PLOT - DATA

Data plotting methods.
"""

__all__ = ['DataXYPlot']

from MLStructFP_benchmarks.ml.utils.plot._utils import get_transparency_from_data, get_thousands_int_dot_sep
from MLStructFP_benchmarks.ml.utils import r2_score
from MLStructFP.utils import save_figure, configure_figure, DEFAULT_PLOT_DPI, \
    DEFAULT_PLOT_FIGSIZE

from matplotlib.ticker import FormatStrFormatter, PercentFormatter
from typing import TYPE_CHECKING, Union, List, Any, Optional
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import statistics
import warnings

# noinspection PyProtectedMember
from pandas.plotting._matplotlib.hist import _grouped_hist, create_subplots, flatten_axes, \
    set_ticks_props

try:
    # noinspection PyProtectedMember
    from pandas.plotting._matplotlib.hist import ABCIndexClass
except ImportError:
    ABCIndexClass = list

if TYPE_CHECKING:
    from MLStructFP_benchmarks.ml.model.core import ModelDataXY

warnings.filterwarnings('ignore', category=FutureWarning)

# Plot configs
plt.show = plt.gcf


def _hist_frame(
        data,
        column=None,
        by=None,
        grid=True,
        xlabelsize=None,
        xrot=None,
        ylabelsize=None,
        titlesize=None,
        yrot=None,
        ax=None,
        sharex=False,
        sharey=False,
        figsize=None,
        layout=None,
        boldtitle=True,
        bins=10,
        **kwds
) -> Any:
    """
    Re-implementation of pandas hist_frame.
    """
    if by is not None:
        axes = _grouped_hist(
            data,
            column=column,
            by=by,
            ax=ax,
            grid=grid,
            figsize=figsize,
            sharex=sharex,
            sharey=sharey,
            layout=layout,
            bins=bins,
            xlabelsize=xlabelsize,
            xrot=xrot,
            ylabelsize=ylabelsize,
            yrot=yrot
        )
        return axes

    if column is not None:
        if not isinstance(column, (list, np.ndarray, ABCIndexClass)):
            column = [column]
        data = data[column]
    # noinspection PyProtectedMember
    data = data._get_numeric_data()
    naxes = len(data.columns)

    if naxes == 0:
        raise ValueError('hist method requires numerical columns, nothing to plot.')

    fig, axes = create_subplots(
        naxes=naxes,
        ax=ax,
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
        figsize=figsize,
        layout=layout
    )
    _axes = flatten_axes(axes)

    # print(data.columns)
    for i, col in enumerate(data.columns):
        ax = _axes[i]
        ax.hist(data[col].dropna().values, bins=bins, log=True)
        titlec = col
        if boldtitle:
            titlec = '$\\bf{' + col + '}$'
        ax.set_title(titlec, fontsize=titlesize)
        ax.grid(grid)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    set_ticks_props(
        axes, xlabelsize=xlabelsize, xrot=xrot, ylabelsize=ylabelsize, yrot=yrot
    )
    fig.subplots_adjust(wspace=kwds.get('fig_wspace', 0.5), hspace=kwds.get('fig_wspace', 0.3))

    return axes


class DataXYPlot(object):
    """
    Data plotting methods.
    """

    _data: 'ModelDataXY'

    def __init__(self, data: 'ModelDataXY') -> None:
        """
        Constructor.

        :param data: Data object
        """
        self._data = data

    def hist(
            self,
            xy: str,
            figsize: int = 3 * DEFAULT_PLOT_FIGSIZE,
            bins: int = 50,
            grid: bool = True,
            drop_columns: Union[List[str], str] = '',
            save: str = '',
            **kwargs
    ) -> None:
        """
        Histogram of data.

        :param xy: Which dataframe
        :param figsize: Size of figure
        :param bins: Number of bins
        :param grid: Display grid
        :param drop_columns: Delete columns
        :param save: Save figure
        :param kwargs: Optional keyword arguments
        """
        plt.figure(figsize=(figsize, figsize), dpi=DEFAULT_PLOT_DPI)
        ax = plt.axes()
        xy = xy.lower()
        data: 'pd.DataFrame' = self._data.get_dataframe(xy=xy)
        if isinstance(drop_columns, list) and len(drop_columns) > 0:
            data = data.drop(columns=drop_columns)
        fsz = kwargs.get('font_size', 13)  # Ticks
        _hist_frame(
            data,
            ax=ax,
            bins=bins,
            figsize=figsize,
            grid=grid,
            log=True,
            titlesize=kwargs.get('title_size', 12),
            xlabelsize=fsz,
            ylabelsize=fsz,
            **kwargs
        )
        save_figure(save, **kwargs)
        plt.show()

    def hist_column(self, xy: str, column: str, bins: int = 25, dpi: int = 100, save: str = '', **kwargs) -> None:
        """
        Histogram of a certain column of data.

        :param xy: Which dataframe
        :param column: Column name
        :param bins: Number of bins
        :param dpi: DPI of output figure
        :param save: Save figure
        :param kwargs: Optional keyword arguments
        """
        data: 'pd.DataFrame' = self._data.get_dataframe(xy=xy)
        if not self._data.has_column(column_name=column):
            raise ValueError(f'Column <{column}> does not exist in dataframe "{xy}"')
        plt.figure(dpi=dpi)
        data.hist(column=column, bins=bins, ax=plt.gca())
        configure_figure(**kwargs)
        save_figure(save, **kwargs)
        plt.show()

    def boxplot_column(self, xy: str, column: str, save: str = '', **kwargs) -> None:
        """
        Boxplot of a certain column of data.

        :param xy: Which dataframe
        :param column: Column name
        :param save: Save figure
        :param kwargs: Optional keyword arguments
        """
        data: 'pd.DataFrame' = self._data.get_dataframe(xy=xy)
        if not self._data.has_column(column_name=column):
            raise ValueError(f'Column <{column}> does not exist in dataframe "{xy}"')
        plt.figure(dpi=DEFAULT_PLOT_DPI)
        data.boxplot(column=column, ax=plt.gca())
        configure_figure(**kwargs)
        save_figure(save, **kwargs)
        plt.show()

    def distribution_column(
            self,
            xy: str,
            column: str,
            square: bool = False,
            labelsize: int = 15,
            save: str = '',
            **kwargs
    ) -> None:
        """
        Distribution of a certain column of data.

        :param xy: Which dataframe
        :param column: Column name
        :param square: Use square plot
        :param labelsize: Size of labels
        :param save: Save figure
        :param kwargs: Optional keyword arguments
        """
        data: 'pd.DataFrame' = self._data.get_dataframe(xy=xy)
        sns.set(style='ticks')
        if not square:
            plt.figure(figsize=(2 * DEFAULT_PLOT_FIGSIZE, DEFAULT_PLOT_FIGSIZE), dpi=DEFAULT_PLOT_DPI)
        else:
            plt.figure(figsize=(DEFAULT_PLOT_FIGSIZE, DEFAULT_PLOT_FIGSIZE), dpi=DEFAULT_PLOT_DPI)
        if not self._data.has_column(column_name=column):
            raise ValueError(f'Column <{column}> does not exist in dataframe "{xy}"')
        sns.distplot(data[column])
        plt.tick_params(labelsize=labelsize)
        plt.title(f'Distribution of {column} - {xy.upper()}')
        configure_figure(**kwargs)
        save_figure(save, **kwargs)
        plt.show()

    # noinspection SpellCheckingInspection
    def correlation_matrix(
            self,
            xy: str,
            title: str = '',
            drop_duplicated: bool = False,
            drop_columns: Union[List[str], str] = '',
            save: str = '',
            **kwargs
    ) -> None:
        """
        Create correlation matrix from the data.

        :param xy: Which dataframe
        :param title: Figure title
        :param drop_duplicated: Drop duplicated data (labeled)
        :param drop_columns: Delete columns
        :param save: Save figure
        :param kwargs: Optional keyword arguments
        """
        # Create the plot
        plt.figure(figsize=(DEFAULT_PLOT_FIGSIZE, DEFAULT_PLOT_FIGSIZE), dpi=DEFAULT_PLOT_DPI)
        ax = plt.axes()
        data: 'pd.DataFrame' = self._data.get_dataframe(xy=xy)
        if not isinstance(drop_columns, list):
            drop_columns = [drop_columns]
        if len(drop_columns) > 0:
            for k in range(len(drop_columns)):
                data = data.drop(columns=drop_columns[k])

        # Delete repeated data
        if drop_duplicated:
            data = data.drop_duplicates()

        # Calculate correlation
        corr = data.corr()
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        if title == '':
            title = 'Correlation Matrix - {1}'.format(title, xy.upper())
        print('Mean:', np.mean(corr.values))
        print('Std:', np.std(corr.values))

        # Apply mask
        with sns.axes_style('white'):
            sns.heatmap(
                corr,
                annot=not kwargs.get('disable_anotation', False),
                annot_kws={'size': 4},
                cbar_kws={
                    'aspect': kwargs.get('cbar_aspect', 50),
                    'pad': 0.02,
                    'shrink': kwargs.get('cbar_shrink', 0.85)
                },
                ax=ax,
                fmt='.0g',
                linewidths=0.25,
                mask=mask,
                square=True,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values
            )

        # Set plot
        if title != 'null':
            ax.set_title(title)
        plt.setp(
            ax.get_xticklabels(),
            ha='right',
            rotation=45,
            rotation_mode='anchor',
            fontsize=kwargs.get('font_size')
        )
        plt.setp(
            ax.get_yticklabels(),
            fontsize=kwargs.get('font_size')
        )
        ylims = ax.get_ylim()
        ax.set_ylim(math.ceil(ylims[0]), math.floor(ylims[1]))
        # configure_figure(**kwargs)
        save_figure(save, **kwargs)
        plt.show()

    def image_df(
            self,
            xy: str,
            column_id: str,
            obj_id: int,
            print_imid: bool = False,
            save: str = '',
            **kwargs
    ) -> None:
        """
        Returns image from a given id.

        :param xy: Which dataframe ID
        :param column_id: Column ID
        :param obj_id: ID value
        :param print_imid: Print image ID
        :param save: Save figure
        :param kwargs: Optional keyword arguments
        """
        im_df = self._data.get_image_data(xy=xy)
        id_col = self._data.get_dataframe(xy=xy, get_id=True)
        assert column_id in id_col.columns, f'Column ID <{column_id}> does not exists on dataframe <{xy}>'
        objindx: 'pd.DataFrame' = id_col[id_col[column_id] == obj_id]
        if len(objindx.index.values) == 0:
            raise ValueError(f'ID <{obj_id}> does not exist in dataframe <{xy}>')
        elif len(objindx.index.values) > 1:
            raise ValueError(f'ID <{obj_id}> has multiple values in dataframe <{xy}>')
        indx = int(objindx.index.values[0])
        im_mat = im_df[indx]
        plt.figure(dpi=DEFAULT_PLOT_DPI)
        plt.imshow(im_mat, cmap='gray')
        plt.xlabel('x $(px)$')
        plt.ylabel('y $(px)$')
        title_imid = ''
        if print_imid:
            # noinspection PyProtectedMember
            title_imid = f'\nImage ID {objindx[self._data._image_col].values[0]} (POS {indx})'
        plt.title(f'Object {column_id} {obj_id} ─ {xy}{title_imid}\n')
        kwargs['cfg_grid'] = False
        configure_figure(**kwargs)
        save_figure(save, **kwargs)
        plt.show()

    def correlation(
            self,
            column: str,
            min_lim: Optional[float] = None,
            max_lim: Optional[float] = None,
            transparency: bool = True,
            data_filter: Optional['pd.Series'] = None,
            xy: str = 'dataset',
            save: str = '',
            **kwargs
    ) -> None:
        """
        Correlation of a certain column from X and Y data sets.

        :param column: Column name
        :param min_lim: Min limit
        :param max_lim: Max limit
        :param transparency: Use transparency
        :param data_filter: Use data filter
        :param xy: Modes of correlation, can be dataset, train, test
        :param save: Save figure
        :param kwargs: Optional keyword arguments
        """
        if min_lim is not None and max_lim is not None:
            assert min_lim < max_lim

        _dfx = ''
        _dfy = ''
        if xy == 'dataset':
            _dfx = 'xdata'
            _dfy = 'ydata'
        elif xy == 'train':
            _dfx = 'xtrain'
            _dfy = 'ytrain'
        elif xy == 'test':
            _dfx = 'xtest'
            _dfy = 'ytest'
        else:
            raise ValueError('Invalid xy mode, valid "dataset", "train" or "test"')

        # Get columns and inverse from scaler
        x = self._data.get_inversed_column_minmax_scaler(xy=_dfx, column_name=column)
        y = self._data.get_inversed_column_minmax_scaler(xy=_dfy, column_name=column)

        if data_filter is not None:
            x = x[data_filter]
            y = y[data_filter]

        print('Total points:', len(x))

        # Create figure
        plt.figure(figsize=(DEFAULT_PLOT_FIGSIZE, DEFAULT_PLOT_FIGSIZE), dpi=DEFAULT_PLOT_DPI)
        if transparency:
            plt.plot(x, y, 'ko', alpha=get_transparency_from_data(x),
                     fillstyle='full', markeredgewidth=0, markersize=4)
        else:
            plt.plot(x, y, 'k.')

        if kwargs.get('extend_xlabel', True):
            plt.xlabel(f"$x_{'{' + xy + '}'}$" + kwargs.get('xlabel', ''))
            plt.ylabel(f"$y_{'{' + xy + '}'}$" + kwargs.get('ylabel', ''))
        else:
            plt.xlabel(kwargs.get('xlabel', ''))
            plt.ylabel(kwargs.get('ylabel', ''))
        if kwargs.get('show_title', True):
            plt.title(column.capitalize() + f' ─ {get_thousands_int_dot_sep(len(x))} samples')
        # plt.plot(_lims,_lims,'r--',linewidth = 2)

        # Plot info text
        text = ''

        # Create linear regression
        if kwargs.get('r2_enabled', True):
            _, _, _, t = r2_score(x, y, origin=False, plot=True, plot_style='r', plot_lw=0.75)
            text += t

        # Create ratio (y/x)
        if kwargs.get('ratio_enabled', True):
            if kwargs.get('ratio_filter_both', False):
                not0mask = (np.abs(x) > kwargs.get('ratio_threshold', 1e-3)) & \
                           (np.abs(y) > kwargs.get('ratio_threshold', 1e-3))
            else:
                not0mask = (np.abs(x) > kwargs.get('ratio_threshold', 1e-3))
            ratio = y[not0mask] / x[not0mask]

            ratio_mean = np.mean(ratio)
            ratio_std = np.std(ratio)
            ratio_mode = statistics.mode(ratio)
            if text != '':
                text += '\n'
            ratio_label = kwargs.get('ratio_label', 'ratio')
            text += f'${ratio_label}={ratio_mean:.03f} \pm {ratio_std:.03f}$'

            # If only plot hist
            if kwargs.get('ratio_hist', False):
                plt.close()
                plt.figure(figsize=(DEFAULT_PLOT_FIGSIZE, DEFAULT_PLOT_FIGSIZE), dpi=DEFAULT_PLOT_DPI)
                plt.hist(ratio, bins=kwargs.get('ratio_hist_bins', 100), weights=np.ones(len(ratio)) / len(ratio))
                xlims = [min(ratio), max(ratio)]
                if kwargs.get('ratio_hist_xlims', None) is None:
                    plt.xlim(xlims)
                else:
                    plt.xlim(kwargs.get('ratio_hist_xlims'))
                print('Limits:', xlims)
                print(f'Mean: {ratio_mean}, Mode: {ratio_mode}')
                plt.xlabel(kwargs.get('xlabel', 'Ratio'))
                plt.ylabel(kwargs.get('ylabel', 'Percentage'))
                plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=kwargs.get('num_decimals', 1)))
                ratio_p = [kwargs.get('ratio_under_perc', 0.5), kwargs.get('ratio_over_perc', 1.5)]
                ratio_under_p = round((len(np.where(ratio < ratio_p[0])[0]) / len(x)) * 100, 2)
                print(f'Percentage of ratio under {ratio_p[0]}: {ratio_under_p}%')
                ratio_over_p = round((len(np.where(ratio > ratio_p[1])[0]) / len(x)) * 100, 2)
                print(f'Percentage of ratio over {ratio_p[1]}: {ratio_over_p}%')
                if kwargs.get('ratio_hist_grid', True):
                    plt.grid()
                leg_position = kwargs.get('legend_position', 0.04)
                assert 0 <= leg_position < 1
                text = f'${ratio_label}={ratio_mean:.03f} \pm {ratio_std:.03f}$\n'
                text += f'$Mode={ratio_mode:.03f}$'
                plt.gca().text(0.46, 1 - leg_position, text, transform=plt.gca().transAxes,
                               fontsize=kwargs.get('reg_fontsize', 10), verticalalignment='top',
                               bbox=dict(facecolor='white', alpha=0.75))
                return

        if text != '':
            leg_position = kwargs.get('legend_position', 0.04)
            assert 0 <= leg_position < 1
            plt.gca().text(leg_position, 1 - leg_position, text, transform=plt.gca().transAxes,
                           fontsize=kwargs.get('reg_fontsize', 12), verticalalignment='top',
                           bbox=dict(facecolor='white', alpha=0.75))

        # Update plot
        if min_lim is None:
            min_lim = min(min(plt.xlim()), min(plt.ylim()))
        if max_lim is None:
            max_lim = max(max(plt.xlim()), max(plt.ylim()))
        if max_lim > min_lim:
            _lims = [min_lim, max_lim]
        else:
            _lims = [max_lim, min_lim]
        plt.xlim(_lims)
        plt.ylim(_lims)
        configure_figure(**kwargs)
        save_figure(save, **kwargs)
        plt.show()

    def assoc_metrics_simple(
            self,
            x: str = 'projectID',
            y: str = 'mean',
            dpi: int = 100,
            save: str = '',
            **kwargs
    ) -> None:
        """
        Plot simple assoc metrics.

        :param x: X column
        :param y: Y column
        :param dpi: Figure dpi
        :param save: Save figure
        :param kwargs: Optional keyword arguments
        """
        plt.figure(dpi=dpi)
        data = self._data.get_metrics()
        plt.plot(data[x], data[y], '.k')
        plt.xlabel(x)
        plt.ylabel(y)
        configure_figure(**kwargs)
        save_figure(save, **kwargs)
        plt.show()

    def assoc_metrics(self, x: str = 'projectID', y: str = 'mean') -> 'go.Figure':
        """
        Plot assoc metrics.

        :param x: X column
        :param y: Y column
        """
        data = self._data.get_metrics()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            fill='none',
            line=dict(color='#000000'),
            mode='markers',
            x=data[x],
            y=data[y],  # mae_thickness
            name='Assoc metrics'
        ))
        assoc_mean = statistics.mean(data['mean'])
        fig.add_trace(go.Scatter(
            fill='none',
            mode='lines',
            x=[min(data[x]), max(data[x])],
            y=[assoc_mean, assoc_mean],
            line=dict(color='#ff0000', dash='dash'),
            name=f'{y.title()}: {assoc_mean:.3f}'
        ))
        fig.update_layout(
            title='Assoc score',
            yaxis_zeroline=False,
            xaxis_zeroline=False
        )
        fig.update_xaxes(title_text=x.title(), hoverformat='.3f')
        fig.update_yaxes(title_text=y.title(), hoverformat='.3f')
        return fig

    def assoc_metric_hist(self, column: str, save: str = '', **kwargs) -> None:
        """
        Assoc hist metric.

        :param column: Column name
        :param save: Save figure
        :param kwargs: Optional keyword arguments
        """
        data = self._data.get_metrics()
        plt.figure(dpi=DEFAULT_PLOT_DPI)
        data.hist(column=column, ax=plt.gca())
        configure_figure(**kwargs)
        save_figure(save, **kwargs)
        plt.show()
