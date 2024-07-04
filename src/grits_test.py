import unittest
from src import grits
import numpy as np
import pprint

class Align1d(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(grits.align_1d([], [], None, return_alignment=False), 0)

    def test_empty_one_side(self):
        self.assertEqual(0, grits.align_1d([], [1, 4, 2], None, return_alignment=False))
        self.assertEqual(0, grits.align_1d([1, 4, 2], [], None, return_alignment=False))

    def test_one_against_one(self):
        self.assertEqual(
            7, grits.align_1d(["a"], ["b"], {"ab": 7}, return_alignment=False)
        )


class Lcs(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(0, grits.lcs(0, 0, None, np.int32))

    def test_empty_one_side(self):
        self.assertEqual(0, grits.lcs(0, 3, None, np.int32))
        self.assertEqual(0, grits.lcs(3, 0, None, np.int32))

    def test_one_against_one(self):
        self.assertEqual(7, grits.lcs(1, 1, lambda x, y: None if x or y else 7, np.int32))

    def test_random(self):
        generator = np.random.default_rng(12345)
        experiments = 1000
        m_exclusive_bound, n_exclusive_bound = 50, 100
        m = generator.integers(0, m_exclusive_bound, size=(experiments,))
        n = generator.integers(0, n_exclusive_bound, size=(experiments,))
        reward_lookup = generator.integers(
            0, 100, size=(experiments, m_exclusive_bound, n_exclusive_bound)
        )
        for experiment_index in range(experiments):
            sequence1 = [(i,) for i in range(m[experiment_index])]
            sequence2 = [(j,) for j in range(n[experiment_index])]
            expected = grits.align_1d(
                sequence1,
                sequence2,
                reward_lookup[experiment_index],
                return_alignment=False,
            )
            actual = grits.fast_align_1d(
                sequence1, sequence2, reward_lookup[experiment_index]
            )
            self.assertEqual(expected, actual)

    def test_random_strings(self):
        generator = np.random.default_rng(12345)
        experiments = 100
        m_exclusive_bound, n_exclusive_bound = 50, 100
        m = generator.integers(0, m_exclusive_bound, size=(experiments,))
        n = generator.integers(0, n_exclusive_bound, size=(experiments,))
        reward_lookup = generator.integers(
            0, 100, size=(experiments, m_exclusive_bound, n_exclusive_bound)
        )
        for experiment_index in range(experiments):
            sequence1 = [("x{}y".format(i),) for i in range(m[experiment_index])]
            sequence2 = [("a{}b".format(j),) for j in range(n[experiment_index])]
            r = {("x{}y".format(i), "a{}b".format(j)): reward_lookup[experiment_index, i, j] for i in range(m[experiment_index]) for j in range(n[experiment_index])}
            expected = grits.align_1d(
                sequence1,
                sequence2,
                r,
                return_alignment=False,
            )
            actual = grits.fast_align_1d(
                sequence1, sequence2, r
            )
            self.assertEqual(expected, actual)
