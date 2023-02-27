"""
MLSTRUCTFP BENCHMARKS - ML - MODEL - CORE - XY IMAGES

Model based on xy + images.
"""

__all__ = ['GenericModelXYImage']

from abc import ABCMeta
from MLStructFP_benchmarks.ml.model.core import GenericModelXY, GenericModelImage

from typing import TYPE_CHECKING, Dict, Any, Optional

if TYPE_CHECKING:
    from MLStructFP_benchmarks.ml.model.core import ModelDataXY


# noinspection PyProtectedMember
class GenericModelXYImage(GenericModelXY, GenericModelImage, metaclass=ABCMeta):
    """
    XY + Image model.
    """

    def __init__(self, data: Optional['ModelDataXY'], name: str, *args, **kwargs) -> None:
        """
        Constructor.

        :param data: Model data
        :param name: Model name
        :param sel_columns_x: Select columns on x data, if empty select all
        :param sel_columns_y: Select columns on y data, if empty select all
        :param img_size: Custom image size (px)
        :param img_channels: Number of channels of the image
        :param args: Optional non-keyword arguments
        :param kwargs: Optional keyword arguments
        """
        GenericModelXY.__init__(self, data=data, name=name, *args, **kwargs)
        GenericModelImage.__init__(self, data=data, name=name, *args, **kwargs)

    def _custom_save_session(self, filename: str, data: dict) -> None:
        """
        See upper doc.
        """
        GenericModelXY._custom_save_session(self, filename, data)
        GenericModelImage._custom_save_session(self, filename, data)

    def _custom_load_session(
            self,
            filename: str,
            asserts: bool,
            data: Dict[str, Any],
            check_hash: bool
    ) -> None:
        """
        See upper doc.
        """
        GenericModelXY._custom_load_session(self, filename, asserts, data, check_hash)
        GenericModelImage._custom_load_session(self, filename, asserts, data, check_hash)
