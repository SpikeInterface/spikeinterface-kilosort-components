import unittest

from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite

from spikeinterface.sorters import run_sorter

from pathlib import Path

from spikeinterface_kilosort_components.kilosort_like_sorter import Kilosort4LikeSorter


class Kilosort4LikeSorterCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = Kilosort4LikeSorter


if __name__ == "__main__":
    test = Kilosort4LikeSorterCommonTestSuite()
    test.cache_folder = Path(__file__).resolve().parents[2] / "cache_folder" / "sorters"
    test.setUp()
    test.test_with_run()
