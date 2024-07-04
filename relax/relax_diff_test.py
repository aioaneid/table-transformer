import unittest
from relax import relax_diff
import numpy as np


class CutoffOneMean(unittest.TestCase):
    def test_empty(self):
        for accurate_sum in (False, True):
            np.testing.assert_equal(
                relax_diff.cutoff(np.empty(0), [-1], accurate_sum=accurate_sum), [-1]
            )
            np.testing.assert_equal(
                relax_diff.cutoff(np.empty(0), [0], accurate_sum=accurate_sum), [0]
            )
            np.testing.assert_equal(
                relax_diff.cutoff(np.empty(0), [1], accurate_sum=accurate_sum), [1]
            )

    def test_one(self):
        for accurate_sum in (False, True):
            np.testing.assert_equal(
                relax_diff.cutoff(np.array([10]), [9], accurate_sum=accurate_sum), [9]
            )
            np.testing.assert_equal(
                relax_diff.cutoff(np.array([10]), [10], accurate_sum=accurate_sum),
                [np.inf],
            )
            np.testing.assert_equal(
                relax_diff.cutoff(np.array([10]), [11], accurate_sum=accurate_sum),
                [np.inf],
            )

    def test_two(self):
        for accurate_sum in (False, True):
            np.testing.assert_equal(
                relax_diff.cutoff(np.array([8, 10]), [7], accurate_sum=accurate_sum),
                [7],
            )
            np.testing.assert_equal(
                relax_diff.cutoff(np.array([8, 10]), [8], accurate_sum=accurate_sum),
                [8],
            )
            np.testing.assert_equal(
                relax_diff.cutoff(np.array([8, 10]), [9], accurate_sum=accurate_sum),
                [np.inf],
            )
            np.testing.assert_equal(
                relax_diff.cutoff(np.array([8, 10]), [10], accurate_sum=accurate_sum),
                [np.inf],
            )


class CutoffMultipleMeans(unittest.TestCase):
    def test_empty(self):
        for accurate_sum in (False, True):
            np.testing.assert_equal(
                relax_diff.cutoff(np.empty(0), [-1, 0, 1], accurate_sum=accurate_sum),
                [-1, 0, 1],
            )

    def test_one(self):
        for accurate_sum in (False, True):
            np.testing.assert_equal(
                relax_diff.cutoff(
                    np.array([10]), [9, 10, 11], accurate_sum=accurate_sum
                ),
                [9, np.inf, np.inf],
            )

    def test_two(self):
        for accurate_sum in (False, True):
            np.testing.assert_equal(
                relax_diff.cutoff(
                    np.array([8, 10]), [7, 8, 9, 10], accurate_sum=accurate_sum
                ),
                [7, 8, np.inf, np.inf],
            )


if __name__ == "__main__":
    unittest.main()
