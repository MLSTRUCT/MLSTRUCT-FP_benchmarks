"""
MLSTRUCT-FP BENCHMARKS - UTILS - DATASET GENERATOR

Generates a standard dataset by applying mutators to a given floor,
the results can be stored to .npz compressed files, or saved to a file.
"""

__all__ = ['FPDatasetGenerator']

from MLStructFP.db import DbLoader
from MLStructFP.utils import make_dirs
from MLStructFP_benchmarks.utils._fp_patch_generator import FloorPatchGenerator

from MLStructFP.db.image import restore_plot_backend
# noinspection PyProtectedMember
from MLStructFP.db.image._rect_photo import RectFloorPhotoFileLoadException

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
                           dx: DeltaPatchType, dy: DeltaPatchType, p: str, r: RotationAnglesType,
                           mba: float) -> Tuple[int, int]:
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
    :param r: Rotation angles
    :param mba: Min binary area
    :return: Number of (added, ignored) patches
    """
    floors = db.floors
    print(f'Processing floor {i + 1}/{len(floors)}')
    gen = FPDatasetGenerator(image_size=isz, patch_size=psz, bw=bw, delta_x=dx, delta_y=dy,
                             min_binary_area=mba)
    # noinspection PyProtectedMember
    nb = gen._process_floor(floors[i], rotation_angles=r, verbose=False)
    gen.export(path=f'{p}{floors[i].id}')
    del gen
    return nb


class FPDatasetGenerator(object):
    """
    Generates a dataset floor plan patches by introducing several mutations.
    """
    _gen: 'FloorPatchGenerator'
    _processed_floor: List[int]

    def __init__(
            self,
            image_size: int,
            patch_size: float,
            bw: bool = True,
            delta_x: DeltaPatchType = (-0.25, 0, 0.25),
            delta_y: DeltaPatchType = (-0.25, 0, 0.25),
            min_binary_area: float = 0
    ) -> None:
        """
        Constructor.

        :param image_size: Image size in px. It must be a power of two
        :param patch_size: Dimensions in (m) to crop the floor plan for x/y-axis
        :param bw: Convert all images to black/white. Recommended as color does not contribute to the plan semantics
        :param delta_x: Delta crops/sliding-window for each patch (from -0.5,-0.5). If None, only iterate in y-axis
        :param delta_y: Delta crops for each patch (from -0.5,-0.5). If both are None, there is only one crop per patch
        :param min_binary_area: Min area for binary image allowed (0-1). If lower, ignores the patch
        """
        self._gen = FloorPatchGenerator(
            image_size=image_size,
            patch_size=patch_size,
            bw=bw,
            delta_x=delta_x,
            delta_y=delta_y,
            min_binary_area=min_binary_area
        )
        self._processed_floor = []

    # noinspection PyProtectedMember
    def process_dataset(
            self,
            db: 'DbLoader',
            path: str,
            rotation_angles: RotationAnglesType = DEFAULT_ROTATION_ANGLES,
            **kwargs) -> List[Tuple[int, int]]:
        """
        Exports a dataset in parallel (each floor is exported indepently).

        :param db: Dataset to process
        :param path: Path to export the data. The floor ID is appended to the path, for example, "abc/cd" => "abc/cd_123"
        :param rotation_angles: Which rotation angles are applied to the floor plan
        :param kwargs: Optional keyword arguments
        :return: List of (added, ignored) patches for each floor in the dataset
        """
        num_proc = kwargs.get('num_thread_processes', 8)
        t0 = time.time()
        num_proc = min(num_proc, cpu_count())
        isz, psz = self._gen._image_size, self._gen._patch_size
        bw, dx, dy = self._gen._bw, self._gen._dx, self._gen._dy
        mba = self._gen._min_binary_area

        # Check angle
        assert isinstance(rotation_angles, (tuple, list)), 'Rotation angles must be a tuple or a list'
        for angle in rotation_angles:
            assert isinstance(angle, (int, float)), f'Each rotation angle must be numeric. Error at: {angle}'

        # Check path
        path_err = ('Path must contain a "/", examples "mypath/" or "mypath/example_". For both examples, '
                    'dataset will be stored as "mypath/file1_binary.npz", or "mypath/example_file1_photo.npz" '
                    'respectively')
        assert '/' in path, path_err

        t: int = len(db.floors)  # Total floors
        if t == 0:
            print('There are no floors. Process finished')
            return []

        print(f'Total floors to compute in parallel: {t}')
        print(f'Using up to {num_proc}/{cpu_count()} CPUs')
        print(f'Using export path: {path}, image size: {isz}px, patch size: {psz}m')
        print(f'Crop delta x: {dx}, delta y: {dy}, black/white: {bw}, min binary area: {mba}')
        print(f'Rotation angles: {rotation_angles}')

        pool = Pool(processes=num_proc)
        results = pool.map(functools.partial(
            _process_fp_dataset_mp, db=db, isz=isz, psz=psz, bw=bw,
            dx=dx, dy=dy, p=path, r=rotation_angles, mba=mba), range(t))
        pool.close()
        pool.join()
        total_time = time.time() - t0
        added, ignored = 0, 0
        for r in results:
            added += r[0]
            ignored += r[1]

        print(f'Pool finished, total time: {datetime.timedelta(seconds=int(total_time))}')
        print(f'Added patches: {added}, ignored: {ignored}. Ratio: {100 * added / (added + ignored):.1f}%')
        restore_plot_backend()
        gc.collect()
        return results

    # noinspection PyProtectedMember
    def _process_floor(
            self,
            floor: 'Floor',
            rotation_angles: RotationAnglesType = DEFAULT_ROTATION_ANGLES,
            **kwargs
    ) -> Tuple[int, int]:
        """
        Process a given floor.

        :param floor: Floor to process
        :param kwargs: Keyword optional arguments
        :param rotation_angles: Which rotation angles are applied to the floor plan
        :return: Number of (added, ignored) patches
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
            try:
                self._gen.process(floor)
            except RectFloorPhotoFileLoadException:
                print(f'[ERROR] Skipping floor ID <{floor.id}> as its image "{floor.image_path}" could not be loaded')
                continue
            added += self._gen._test_last_added
            ignored += len(self._gen._test_ignored_patches)
            if verbose:
                print('OK')

        floor.mutate(a, sx, sy)  # Rollback
        if verbose:
            print(f'\tFinished in {time.time() - t0:.2f} seconds. Added: {added}, ignored: {ignored}')
        self._processed_floor.append(floor.id)

        return added, ignored

    # noinspection PyProtectedMember
    def export(self, path: str) -> None:
        """
        Export images to file. After export, the object is cleared.

        :param path: Path to export the data, which is extended with _binary and _photo
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
            np.savez_compressed(fn, data=np.array(image_list, dtype='uint8'))  # .npz

        save_list(f'{path}_binary', self._gen._gen_binary.patches)
        save_list(f'{path}_photo', self._gen._gen_photo.patches)
        self._processed_floor.clear()
        self._gen.clear()
