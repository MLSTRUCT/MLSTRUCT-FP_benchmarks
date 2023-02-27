"""
MLSTRUCTFP BENCHMARKS - ML - UTILS - MATH

Math utils.
"""

__all__ = [
    'filter_xylim',
    'scale_array_to_range',
    'r2_score'
]

import numpy as np

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from typing import List, Union, Tuple, Optional

NumberType = Union[float, int]
VectorType = Union[List[NumberType], Tuple[NumberType, ...], 'np.ndarray']
XYLimType = Optional[Union[List[NumberType], Tuple[NumberType, NumberType]]]


def filter_xylim(
        x: VectorType,
        y: VectorType,
        xlim: XYLimType = None,
        ylim: XYLimType = None
) -> Tuple[VectorType, VectorType]:
    """
    Filter xy values.

    :param x: X vector
    :param y: Y vector
    :param xlim: X limits (min, max)
    :param ylim: Y limits (min, max)
    :return: Filtered x, y
    """
    if xlim is not None or ylim is not None:
        x_ = x
        y_ = y
        x = []
        y = []
        for i in range(len(x_)):
            if xlim is not None and ylim is None:
                if xlim[0] <= x_[i] <= xlim[1]:
                    x.append(x_[i])
                    y.append(y_[i])
            elif xlim is None and ylim is not None:
                if ylim[0] <= y_[i] <= ylim[1]:
                    x.append(x_[i])
                    y.append(y_[i])
            elif xlim is not None and ylim is not None:
                if xlim[0] <= x_[i] <= xlim[1] and ylim[0] <= y_[i] <= ylim[1]:
                    x.append(x_[i])
                    y.append(y_[i])
    return x, y


def r2_score(
        x: VectorType,
        y: VectorType,
        origin: bool,
        rf: int = 4,
        plot: bool = False,
        plot_style: str = 'r--',
        plot_lw: float = 1.5
) -> Tuple[float, float, float, str]:
    """
    Returns R² factor of a linear interpolation f(x) = A*x+b

    :param x: X vector
    :param y: Y vector
    :param origin: R² fit from origin position
    :param rf: Rounding factor for plot text
    :param plot: Plots fit
    :param plot_style: Plot style
    :param plot_lw: Plot linewidth
    :return: R², A, b, text for plot label
    """
    m = LinearRegression(fit_intercept=not origin)
    if isinstance(x, (tuple, list)):
        x = np.array([[i] for i in x])
    if isinstance(y, (tuple, list)):
        y = np.array([[i] for i in y])
    m.fit(x, y)
    y_hat = x * m.coef_ + m.intercept_
    coef = m.coef_[0][0]
    intercept = 0 if origin else m.intercept_[0]
    r2 = m.score(x, y)  # Same as sklearn.metrics.r2_score(y, y_hat)
    if origin:
        text = f'$y={round(coef, rf):0.3f}\;x$\n$R^2 = {round(r2, rf)}$'
    else:
        text = f'$y={round(coef, rf):0.3f}\;x{round(intercept, rf):+0.3f}$\n$R^2 = {round(r2, rf)}$'
    if plot:
        plt.plot([min(x), max(x)], [min(y_hat), max(y_hat)], plot_style, lw=plot_lw)
    return r2, m.coef_, m.intercept_, text


def scale_array_to_range(
        array: 'np.ndarray',
        to: Tuple[Union[int, float], Union[int, float]],
        dtype: Optional[str]
) -> 'np.ndarray':
    """
    Scale array to range.

    :param array: Array to scale
    :param to: Range to scale
    :param dtype: Cast to data type
    """
    assert len(to) == 2, 'Scale must have 2 elements (min,max)'
    assert to[1] > to[0]
    _min = float(to[0])
    _max = float(to[1])

    a_min: float = np.min(array)
    a_max: float = np.max(array)
    if a_min == _min and a_max == _max and _min != _max:
        return array

    if dtype is None:
        dtype = 'float64'  # Normal conversion

    new_array = np.interp(array, (a_min, a_max), to).astype(dtype)
    return new_array
