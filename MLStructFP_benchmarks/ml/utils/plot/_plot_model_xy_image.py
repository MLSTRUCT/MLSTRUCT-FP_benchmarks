"""
MLSTRUCTFP BENCHMARKS - ML - MODEL - UTILS - PLOT - MODEL XY + IMAGE

Plot model xy and image.
"""

__all__ = ['GenericModelPlotXYImage']

from MLStructFP_benchmarks.ml.utils.plot._plot_model_image import GenericModelPlotImage
from MLStructFP_benchmarks.ml.utils.plot._plot_model_xy import GenericModelPlotXY

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from MLStructFP_benchmarks.ml.model.core import GenericModelXYImage


class GenericModelPlotXYImage(GenericModelPlotImage, GenericModelPlotXY):
    """
    Model XY plot.
    """
    _model: 'GenericModelXYImage'

    def __init__(self, model: 'GenericModelXYImage') -> None:
        """
        Constructor.

        :param model: Model object
        """
        GenericModelPlotImage.__init__(self, model)
        GenericModelPlotXY.__init__(self, model)
