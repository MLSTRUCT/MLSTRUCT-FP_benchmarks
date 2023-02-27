"""
MLSTRUCTFP BENCHMARKS - TEST - ML

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
        data.plot_image_example_id(data.load_part(1, False), 0, show=False)
        self.assertEqual(data._parts, [302, 748, 848, 966, 1058, 1059, 1060])

    def test_session(self) -> None:
        """
        Test load/save session.
        """
        data = DataFloorPhoto(self._out, shuffle_parts=True)
        data.save_session('.session/data')
        data2 = load_floor_photo_data_from_session('.session/data')
        self.assertEqual(data._parts, data2._parts)