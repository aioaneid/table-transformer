import unittest
from relax import relax_constraints
import numpy as np


class MinMaxIob1d(unittest.TestCase):
    def test_empty_intersection(self):
        np.testing.assert_equal(
            relax_constraints.min_max_iob_1d_interval_limit_interval(
                relax_constraints.IntervalLimit(hole=[1, 4], outer=[0, 5]), [-2, -1]
            ),
            [0, 0],
        )
        np.testing.assert_equal(
            relax_constraints.min_max_iob_1d_interval_limit_interval(
                relax_constraints.IntervalLimit(hole=[1, 4], outer=[0, 5]), [-2, 0]
            ),
            [0, 0],
        )
        np.testing.assert_equal(
            relax_constraints.min_max_iob_1d_interval_limit_interval(
                relax_constraints.IntervalLimit(hole=[1, 4], outer=[0, 5]), [5, 7]
            ),
            [0, 0],
        )

    def test_containment(self):
        np.testing.assert_equal(
            relax_constraints.min_max_iob_1d_interval_limit_interval(
                relax_constraints.IntervalLimit(hole=[1, 4], outer=[0, 5]), [0, 5]
            ),
            [1, 1],
        )
        np.testing.assert_equal(
            relax_constraints.min_max_iob_1d_interval_limit_interval(
                relax_constraints.IntervalLimit(hole=[1, 4], outer=[0, 5]), [-1, 6]
            ),
            [1, 1],
        )

    def test_random(self):
        generator = np.random.default_rng(12345)
        limits = generator.uniform(0, 1, size=(13, 4))
        limits.sort(axis=-1)
        interval2 = generator.uniform(0, 1, size=(17, 2))
        interval2.sort(axis=-1)
        for limit_row in limits:
            low = generator.uniform(limit_row[0], limit_row[1], size=(19))
            upper = generator.uniform(limit_row[2], limit_row[3], size=(23))
            for interval2_row in interval2:
                actual = relax_constraints.min_max_iob_1d_interval_limit_interval(
                    relax_constraints.IntervalLimit(
                        hole=limit_row[1:3], outer=limit_row[0::3]
                    ),
                    interval2_row,
                )
                for l in low:
                    for u in upper:
                        iob = relax_constraints.iob_1d([l, u], interval2_row)
                        assert 0 <= actual[0] <= iob <= actual[1] <= 1, (
                            "l:",
                            l,
                            "u:",
                            u,
                            "interval2_row:",
                            interval2_row,
                            "iob:",
                            iob,
                            "actual:",
                            actual,
                        )

        np.testing.assert_equal(
            relax_constraints.min_max_iob_1d_interval_limit_interval(
                relax_constraints.IntervalLimit(hole=[1, 4], outer=[0, 5]), [0, 5]
            ),
            [1, 1],
        )
        np.testing.assert_equal(
            relax_constraints.min_max_iob_1d_interval_limit_interval(
                relax_constraints.IntervalLimit(hole=[1, 4], outer=[0, 5]), [-1, 6]
            ),
            [1, 1],
        )


def batched_iob_1d(interval, lo, hi):
    return np.maximum(
        0,
        np.minimum(interval[1], hi) - np.maximum(interval[0], lo),
    ) / (interval[1] - interval[0])


class Iob1NonIncreasing(unittest.TestCase):
    def test_random(self):
        generator = np.random.default_rng(239727)
        interval1 = generator.uniform(0, 1, size=(17, 2))
        interval1 = interval1[(interval1[:, 1] - interval1[:, 0]) != 0, :]
        interval1.sort(axis=-1)
        interval2 = generator.uniform(0, 1, size=(19, 2))
        interval2.sort(axis=-1)
        interval2 = interval2[(interval2[:, 1] - interval2[:, 0]) != 0, :]
        x = generator.uniform(0, 1, size=(11,))
        lo = generator.uniform(0, x, size=(171, len(x)))
        hi = generator.uniform(x, 1, size=(191, len(x)))
        proved, counter_example_found, counter_example_not_found = 0, 0, 0
        for i1 in interval1:
            for i2 in interval2:
                for i, xi in enumerate(x):
                    assert (lo[:, i] <= xi).all()
                    assert (xi <= hi[:, i]).all()
                    iob1 = batched_iob_1d(i1, lo[:, i][:, None], hi[:, i][None, :])
                    iob2 = batched_iob_1d(i2, lo[:, i][:, None], hi[:, i][None, :])
                    if relax_constraints.iob_1d_non_increasing(xi, i1, i2):
                        proved += 1
                        np.testing.assert_array_less(iob2, iob1 + 1e-12)
                    else:
                        if (iob1 >= iob2).all():
                            counter_example_not_found += 1
                        else:
                            counter_example_found += 1
        print(
            "proved: {} counter_example_found: {} counter_example_not_found: {}".format(
                proved, counter_example_found, counter_example_not_found
            )
        )


if __name__ == "__main__":
    unittest.main()
