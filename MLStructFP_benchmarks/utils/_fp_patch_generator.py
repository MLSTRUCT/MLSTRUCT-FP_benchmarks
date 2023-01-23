"""
MLSTRUCTFP BENCHMARKS - UTILS - PATCH GENERATOR

Generates an image patch for a given set of floor plans. The image results
are saved as (x, y) tuples, where x stands for the floor plan image, and y
for the binary one obtained directly from the rectangles.

The patches are computed by setting a patch size, fixed for x/y axes, thus,
only square-images are valid. Then it proposes a set of patches, where for
each one several images are obtained using a displacement which cannot exceed
50%. To save the results, the object stores the images in an efficient data
storage object npz, which can be re-loaded to train the models.
"""

__all__ = ['FloorPatchGenerator']

from MLStructFP.db import Floor
from MLStructFP.db.image import RectBinaryImage, RectFloorPhoto
from MLStructFP.utils import *

import matplotlib.pyplot as plt
import math
import numpy as np

from typing import List, Tuple, Optional
from warnings import warn

PatchRectType = List[Tuple[int, bool, float, float, float, float]]


class FloorPatchGenerator(object):
    """
    Patch generator.
    """
    _dx: List[float]
    _dy: List[float]
    _gen_binary: 'RectBinaryImage'
    _gen_photo: 'RectFloorPhoto'
    _img_size: int
    _min_binary_area: float
    _patch_size: float
    _x: List['np.ndarray']
    _y: List['np.ndarray']

    def __init__(
            self,
            image_size: int,
            patch_size: float,
            delta_x: Optional[List[float]] = None,
            delta_y: Optional[List[float]] = None,
            min_binary_area: float = 0.05
    ) -> None:
        """
        Constructor.

        :param image_size: Image size in px. Must be a power of 2
        :param patch_size: Dimension in (m) to crop the floor plan for x/y-axis
        :param delta_x: Delta crops/sliding-window for each patch (from -0.5,-0.5). If None, only iterate in y-axis
        :param delta_y: Delta crops for each patch (from -0.5,-0.5). If both are None, there is only 1 crop per patch
        :param min_binary_area: Min area for binary image allowed (0-1). If lower, ignores the patch
        """
        assert patch_size > 0, 'patch size cannot be negative'
        if delta_x is None:
            delta_x = [0]
        if delta_y is None:
            delta_y = [0]
        assert isinstance(delta_x, list), 'delta x must be an increasing list of float values between -0.5 to 0.5'
        assert isinstance(delta_y, list), 'delta y must be an increasing list of float values between -0.5 to 0.5'
        lx, ly = len(delta_x), len(delta_y)
        assert lx * ly != 0, 'deltas for x and y axis must have at least 1 item'
        if 0 not in delta_x:
            warn('Delta x must contain 0, which was added automatically')
            delta_x.append(0)
        if 0 not in delta_y:
            warn('Delta y must contain 0, which was added automatically')
            delta_y.append(0)
        for i in range(lx):
            assert isinstance(delta_x[i], (int, float))
            delta_x[i] = float(delta_x[i])
            assert 0 <= abs(delta_x[i]) <= 0.5, 'each delta on x-axis must be +-0.5 max'
            if i < lx - 1:
                assert delta_x[i] < delta_x[i + 1], 'delta x vector must be increasing'
        for j in range(ly):
            assert isinstance(delta_y[j], (int, float))
            delta_y[j] = float(delta_y[j])
            assert 0 <= abs(delta_y[j]) <= 0.5, 'each delta on y-axis must be +-0.5 max'
            if j < ly - 1:
                assert delta_y[j] < delta_y[j + 1], 'delta y vector must be increasing'
        assert 0 <= min_binary_area < 1, 'min area must be between 0 and 1. A zero-value accept all patches'
        self._dx = delta_x
        self._dy = delta_y
        self._gen_binary = RectBinaryImage(image_size_px=image_size)
        self._gen_photo = RectFloorPhoto(image_size_px=image_size)
        self._min_binary_area = min_binary_area
        self._patch_size = patch_size
        self._x = []
        self._y = []

    def process(self, floor: 'Floor') -> None:
        """
        Process a given floor.

        :param floor: Floor to process
        """
        print(self._make_patches(floor))

    def _make_patches(self, floor: 'Floor', apply_delta: bool = True) -> PatchRectType:
        """
        Make patches.

        :param floor: Floor object
        :param apply_delta: Apply dx, dy
        :return: List of patches [(number, xmin, xmax, ymin, ymax), ...]
        """
        patches = []
        bb = floor.bounding_box
        nx = math.ceil((bb.xmax - bb.xmin) / self._patch_size)
        ny = math.ceil((bb.ymax - bb.ymin) / self._patch_size)
        n = 1
        for i in range(nx):
            for j in range(ny):
                for dx in self._dx:
                    for dy in self._dy:
                        origin = dx == 0 and dy == 0
                        if not apply_delta and not origin:
                            continue
                        patches.append((
                            n,
                            origin,
                            bb.xmin + self._patch_size * (i + dx),  # x-min
                            bb.xmin + self._patch_size * (i + 1 + dx),  # x-max
                            bb.ymin + self._patch_size * (j + dy),  # y-min
                            bb.ymin + self._patch_size * (j + 1 + dy)  # y-max
                        ))
                        n += 1
        return patches

    def plot_patches(
            self,
            floor: 'Floor',
            photo: bool = False,
            patches: bool = True,
            rect: bool = True
    ) -> None:
        """
        Plot the patches of a given floor.

        :param floor: Floor to plot
        :param patches: Add patches
        :param photo: Add photo crops
        :param rect: Plot rects
        """
        ax: 'plt.Axes'
        fig, ax = plt.subplots(dpi=DEFAULT_PLOT_DPI)
        ax.set_aspect('equal')
        ax.set_facecolor('#000000')
        for r in floor.rect:
            r.plot_matplotlib(ax, color='#000000' if not photo else '#ffffff')
        if patches:
            for p in self._make_patches(floor, apply_delta=True):
                n, origin, xmin, xmax, ymin, ymax = p
                ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], '-' if origin else '--',
                        linewidth=1.5 if origin else 0.4, color='#ff0000' if origin else '#0000ff')
        if photo:
            lim_x = ax.get_xlim()
            lim_y = ax.get_ylim()
            for p in self._make_patches(floor, apply_delta=False):  # Add images
                _, _, xmin, xmax, ymin, ymax = p
                plt.imshow(self._gen_photo.make_region(xmin, xmax, ymin, ymax, floor)[1],
                           extent=[xmin, xmax, ymin, ymax], origin='upper')
            plt.xlim(lim_x)
            plt.ylim(lim_y)
        if rect:
            for r in floor.rect:
                r.plot_matplotlib(ax, color='#000000' if not photo else '#ff00ff')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        configure_figure(cfg_grid=False)
