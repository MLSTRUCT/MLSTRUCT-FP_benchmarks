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
        patchgen = FloorPatchGenerator(image_size=64, patch_size=10)

        patchgen.process(db.floors[0])
        patchgen.plot_patches(db.floors[0])
