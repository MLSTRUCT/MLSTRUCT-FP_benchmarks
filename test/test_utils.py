"""
MLSTRUCTFP BENCHMARKS - TEST - UTILS

Test utils.
"""

import os
import unittest

from MLStructFP.db import DbLoader
from MLStructFP_benchmarks.utils import FloorPatchGenerator

DB_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'fp.json')


class UtilsTest(unittest.TestCase):

    def test_fp_patch_generator(self) -> None:
        """
        Test patch generator.
        """
        db = DbLoader(DB_PATH)
        patchgen = FloorPatchGenerator(bw=True, image_size=64, patch_size=10)

        floor = db.floors[0]
        patchgen.process(floor)
        self.assertEqual(len(patchgen._make_patches(floor)), 40)
        self.assertEqual(patchgen._test_ignored_patches,
                         [5, 16, 17, 18, 21, 22, 23, 26, 27, 28, 31, 32, 33, 35, 36, 37, 38, 40])

        # Test plots
        patchgen.plot_patches(floor)
        patchgen.plot_patch(0)
        self.assertEqual(len(patchgen._patch_photo), 22)
