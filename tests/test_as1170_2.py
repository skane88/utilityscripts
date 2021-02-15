"""
Contains some tests for the AS1170.2 module.
"""

from pathlib import Path

import toml

FILE_PATH = Path(__file__)
TEST_DATA_PATH = FILE_PATH.parent / Path("test_as1170_2.toml")
TEST_DATA = toml.load(TEST_DATA_PATH)

pass
