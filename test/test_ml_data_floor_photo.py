"""
MLSTRUCTFP BENCHMARKS - TEST - ML

Test data floor photo load.
"""

import os
import unittest

from MLStructFP_benchmarks.ml.model.core import DataFloorPhoto

DB_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'fp.json')


class UtilsTest(unittest.TestCase):

    def test_data_parts(self) -> None:
        """
        Test patch generator.
        """
        data = DataFloorPhoto('.out/')

