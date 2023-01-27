"""
MLSTRUCTFP BENCHMARKS - UTILS - DATASET GENERATOR

Generates a standard dataset by applying mutators to a given floor,
the results can be stored to .npz compressed files, or saved to a file.
"""

__all__ = ['FPDatasetGenerator']

from MLStructFP.db import DbLoader
from MLStructFP.utils import make_dirs
from MLStructFP_benchmarks.utils._fp_patch_generator import FloorPatchGenerator

import datetime
import functools
import gc
import numpy as np
import time

from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from MLStructFP.db import Floor

DEFAULT_ROTATION_ANGLES = (0, 45, 90, 135, 180, 225, 270, 315)

DeltaPatchType = Optional[Union[List[float], Tuple[float, ...]]]
RotationAnglesType = Union[Tuple[int, ...], List[int]]


def _process_fp_dataset_mp(i: int, db: 'DbLoader', isz: int, psz: float, bw: bool,
                           dx: DeltaPatchType, dy: DeltaPatchType, p: str, c: bool, r: RotationAnglesType) -> None:
    """
    Process FP in parallel.

    :param i: ID of the floor plan to process
    :param db: Database object
    :param isz: Image size (px)
    :param psz: Patch size (m)
    :param bw: Use black/white
    :param dx: Delta crops/sliding-window for each patch on x-axis
    :param dy: Delta crops/sliding-window for each patch on y-axis
    :param p: Export path
    :param c: Use compressed export
    :param r: Rotation angles
    """
    floors = db.floors
    print(f'Processing floor {i + 1}/{len(floors)}')
    gen = FPDatasetGenerator(image_size=isz, patch_size=psz, bw=bw, delta_x=dx, delta_y=dy)
    gen.process_floor(floors[i], rotation_angles=r, verbose=False)
    gen.export(path=f'{p}_{floors[i].id}', compressed=c)
    del gen


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
            delta_x: DeltaPatchType = (-0.25, 0, 0.25),
            delta_y: DeltaPatchType = (-0.25, 0, 0.25)
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

    def process_dataset(
            self,
            db: 'DbLoader',
            path: str,
            compressed: bool = True,
            rotation_angles: RotationAnglesType = DEFAULT_ROTATION_ANGLES,
            **kwargs) -> None:
        """
        Exports a dataset in parallel (each floor is exported indepently).

        :param db: Dataset to process
        :param path: Path to export the data. The floor ID is appended to the path, for example, "abc/cd" => "abc/cd_123"
        :param compressed: Save compressed file
        :param rotation_angles: Which rotation angles are applied to the floor plan
        :param kwargs: Optional keyword arguments
        """
        num_proc = kwargs.get('num_thread_processes', 8)
        t0 = time.time()
        num_proc = min(num_proc, cpu_count())

        t = len(db.floors)
        print(f'Total floors to compute in parallel: {t}')
        print(f'Using up to {num_proc}/{cpu_count()} CPUs')
        print(f'Using export path: {path}, compressed: {compressed}')
        pool = Pool(processes=num_proc)
        # noinspection PyProtectedMember
        pool.map(functools.partial(_process_fp_dataset_mp, db=db, isz=self._gen._image_size, psz=self._gen._patch_size, bw=self._gen._bw,
                                   dx=self._gen._dx, dy=self._gen._dy, p=path, c=compressed, r=rotation_angles), range(t))
        pool.close()
        pool.join()
        total_time = time.time() - t0
        print(f'Pool finished, total time: {datetime.timedelta(seconds=int(total_time))}')
        gc.collect()

    # noinspection PyProtectedMember
    def process_floor(self, floor: 'Floor', rotation_angles: RotationAnglesType = DEFAULT_ROTATION_ANGLES, **kwargs) -> int:
        """
        Process a given floor.

        :param floor: Floor to process
        :param kwargs: Keyword optional arguments
        :param rotation_angles: Which rotation angles are applied to the floor plan
        :returns: Number of added patches
        """
        if floor.id in self._processed_floor:
            raise ValueError(f'Floor ID {floor.id} already processed')
        verbose = kwargs.get('verbose', True)
        if verbose:
            print(f'Processing floor ID {floor.id}')

        added, ignored, t0 = 0, 0, time.time()
        a, sx, sy = floor.mutator_angle, floor.mutator_scale_x, floor.mutator_scale_y
        for angle in rotation_angles:
            if verbose:
                print(f'\tGenerating patches with angle={angle} ... ', end='')
            floor.mutate(angle)
            self._gen.process(floor)
            added += self._gen._test_last_added
            ignored += len(self._gen._test_ignored_patches)
            if verbose:
                print('OK')

        floor.mutate(a, sx, sy)  # Rollback
        if verbose:
            print(f'\tFinished in {time.time() - t0:.2f} seconds. Added: {added}, ignored: {ignored}')
        self._processed_floor.append(floor.id)

        return added

    # noinspection PyProtectedMember
    def export(self, path: str, compressed: bool = True) -> None:
        """
        Export images to file. After export, the object is cleared.

        :param path: Path to export the data, which is extended with _binary and _photo
        :param compressed: Save compressed file
        """

        def save_list(fn: str, image_list: List['np.ndarray']) -> None:
            """
            Save numpy list to file.

            :param fn: Filename
            :param image_list: List of images
            """
            make_dirs(fn)
            if len(image_list) == 0:
                return
            if compressed:
                np.savez_compressed(fn, data=np.array(image_list, dtype='uint8'))  # .npz
            else:
                np.save(fn, np.array(image_list, dtype='uint8'))  # .npy

        save_list(f'{path}_binary', self._gen._patch_binary)
        save_list(f'{path}_photo', self._gen._patch_photo)
        self._processed_floor.clear()
        self._gen.clear()
