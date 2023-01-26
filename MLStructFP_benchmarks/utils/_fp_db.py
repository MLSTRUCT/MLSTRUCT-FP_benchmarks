"""
MLSTRUCTFP BENCHMARKS - UTILS - DATASET GENERATOR

Generates a standard dataset by applying mutators to a given floor,
the results can be stored to .npz compressed files, or saved to a file.
"""

__all__ = ['FPDatasetGenerator']

from MLStructFP_benchmarks.utils._fp_patch_generator import FloorPatchGenerator

import numpy as np
import time

from typing import List, Tuple, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from MLStructFP.db import Floor


class FPDatasetGenerator(object):
    """
    Generates a dataset by creatig floor plan patches by introducing several
    mutations.
    """
    _gen: 'FloorPatchGenerator'
    _processed_floor: List[int]

    def __init__(
            self,
            image_size: int,
            patch_size: float,
            bw: bool = True,
            delta_x: Optional[Union[List[float], Tuple[float, ...]]] = (-0.25, 0, 0.25),
            delta_y: Optional[Union[List[float], Tuple[float, ...]]] = (-0.25, 0, 0.25)
    ) -> None:
        """
        Constructor.

        :param image_size: Image size in px. Must be a power of 2
        :param patch_size: Dimension in (m) to crop the floor plan for x/y-axis
        :param bw: Convert all images to black/white. Recomended as color does not contribute to the plan semantics
        :param delta_x: Delta crops/sliding-window for each patch (from -0.5,-0.5). If None, only iterate in y-axis
        :param delta_y: Delta crops for each patch (from -0.5,-0.5). If both are None, there is only 1 crop per patch
        """
        self._gen = FloorPatchGenerator(
            image_size=image_size,
            patch_size=patch_size,
            bw=bw,
            delta_x=delta_x,
            delta_y=delta_y
        )
        self._processed_floor = []

    # noinspection PyProtectedMember
    def process(self, floor: 'Floor') -> int:
        """
        Process a given floor.

        :param floor: Floor to process
        :returns: Number of added patches
        """
        if floor.id in self._processed_floor:
            raise ValueError(f'Floor ID {floor.id} already processed')
        print(f'Processing floor ID {floor.id}')

        added, ignored, t0 = 0, 0, time.time()
        a, sx, sy = floor.mutator_angle, floor.mutator_scale_x, floor.mutator_scale_y
        for angle in (0, 45, 90, 135, 180, 225, 270, 315):
            print(f'\tGenerating patches with angle={angle} ... ', end='')
            floor.mutate(angle)
            self._gen.process(floor)
            added += self._gen._test_last_added
            ignored += len(self._gen._test_ignored_patches)
            print('OK')

        floor.mutate(a, sx, sy)  # Rollback
        print(f'\tFinished in {time.time() - t0:.2f} seconds. Added: {added}, ignored: {ignored}')
        self._processed_floor.append(floor.id)

        return added

    # noinspection PyProtectedMember
    def export(self, path: str, compressed: bool = True) -> None:
        """
        Export images to file. After export, the object is clered

        :param path: Path to export the data, which is extended with _binary and _photo
        :param compressed: Save compressed file
        """

        def save_list(fn: str, image_list: List['np.ndarray']) -> None:
            """
            Save numpy list to file.

            :param fn: Filename
            :param image_list: List of images
            """
            if compressed:
                np.savez_compressed(fn, data=np.array(image_list, dtype='uint8'))  # .npz
            else:
                np.save(fn, np.array(image_list, dtype='uint8'))  # .npy

        save_list(f'{path}_binary', self._gen._patch_binary)
        save_list(f'{path}_photo', self._gen._patch_photo)
        self._processed_floor.clear()
        self._gen.clear()
