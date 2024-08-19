"""
MLSTRUCT-FP BENCHMARKS - TEST - ML

Test data floor photo load.
"""

import os
import unittest

from MLStructFP.db import DbLoader
from MLStructFP_benchmarks.ml.model.core import DataFloorPhoto, load_floor_photo_data_from_session
from MLStructFP_benchmarks.utils import FPDatasetGenerator

DB_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'fp.json')


class UtilsTest(unittest.TestCase):
    _out: str  # Out path

    def setUp(self) -> None:
        self._out = '.out_hr/'
        if not os.path.isdir(self._out):
            print('Generating database')
            FPDatasetGenerator(128, 10).process_dataset(DbLoader(DB_PATH), self._out, rotation_angles=(0,))

    def test_data_parts(self) -> None:
        """
        Test patch generator.
        """
        data = DataFloorPhoto(self._out, shuffle_parts=False)
        self.assertEqual(data.total_parts, 7)
        self.assertEqual(data.get_image_shape(), (128, 128, 1))

        # Get part
        self.assertEqual(data._parts, [302, 748, 848, 966, 1058, 1059, 1060])
        p = data.load_part(1, False)
        for t in ('binary', 'photo'):
            self.assertEqual(p[t].shape, (90, 128, 128, 1))

        # Plot
        data.plot_image_example_id(p, 0, show=False)

    def test_session(self) -> None:
        """
        Test load/save session.
        """
        data = DataFloorPhoto(self._out, shuffle_parts=True)
        data.save_session('.session/data')
        data2 = load_floor_photo_data_from_session('.session/data')
        self.assertEqual(data._parts, data2._parts)

        data.assemble_train_test(0.7)
        data.save_session('.session/data')
        data2 = load_floor_photo_data_from_session('.session/data')
        data._split = data2._split

    def test_split(self) -> None:
        """
        Test train/test split.
        """
        data = DataFloorPhoto(self._out, shuffle_parts=True).assemble_train_test(0.7)
        self.assertEqual(data.total_parts, 1)  # Because train/test was defined
        self.assertEqual(data.total_images, 817)
        self.assertEqual(data.train_split, 0.57)
        tr_s = 0
        for i in data._split[0]:
            tr_s += data.load_part(data._parts.index(i) + 1, ignore_split=True)['binary'].shape[0]
        self.assertEqual(tr_s, data.load_part(1)['binary'].shape[0])
