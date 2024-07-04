import unittest
from relax import relax
import numpy as np


def old_adjust_box_edges(box, multiplier_and_constant, *, min_deviation):
    factor, constant = multiplier_and_constant
    return (
        (box[0] / 2 + box[2] / 2)
        - max(min_deviation / 2, box[2] / 2 - box[0] / 2 + constant / 2) * factor,
        (box[1] / 2 + box[3] / 2)
        - max(min_deviation / 2, box[3] / 2 - box[1] / 2 + constant / 2) * factor,
        (box[0] / 2 + box[2] / 2)
        + max(min_deviation / 2, box[2] / 2 - box[0] / 2 + constant / 2) * factor,
        (box[1] / 2 + box[3] / 2)
        + max(min_deviation / 2, box[3] / 2 - box[1] / 2 + constant / 2) * factor,
    )


class AdjustBoxEdges(unittest.TestCase):
    def test_finite_random(self):
        generator = np.random.default_rng(12345)
        box_source = generator.uniform(0, 1000, size=(19, 4))
        boxes = np.empty_like(box_source)
        boxes[:, :2] = np.minimum(box_source[:, :2], box_source[:, 2:])
        boxes[:, 2:] = np.maximum(box_source[:, :2], box_source[:, 2:])
        multiplier_and_constant = generator.uniform(-10, 10, size=(23, 2))
        min_deviation = generator.uniform(0, 10, size=(31))
        for box in boxes:
            for multiplier_and_constant_row in multiplier_and_constant:
                for min_deviation_row in min_deviation:
                    expected = old_adjust_box_edges(
                        box,
                        multiplier_and_constant_row,
                        min_deviation=min_deviation_row,
                    )
                    actual = relax.adjust_box_edges(
                        box,
                        (multiplier_and_constant_row,) * 4,
                        min_deviation=min_deviation_row,
                    )
                    np.testing.assert_allclose(expected, actual)

    def test_random_finite_exact(self):
        generator = np.random.default_rng(12345)
        box_source = generator.uniform(0, 1000, size=(19, 4))
        boxes = np.empty_like(box_source)
        boxes[:, :2] = np.minimum(box_source[:, :2], box_source[:, 2:])
        boxes[:, 2:] = np.maximum(box_source[:, :2], box_source[:, 2:])
        for box in boxes:
            actual = relax.adjust_box_edges(
                box,
                ((1, 0),) * 4,
                min_deviation=0,
            )
            np.testing.assert_array_equal(box, actual)

    def test_random_infinite_factor(self):
        generator = np.random.default_rng(12345)
        box_source = generator.uniform(0, 1000, size=(19, 4))
        boxes = np.empty_like(box_source)
        boxes[:, :2] = np.minimum(box_source[:, :2], box_source[:, 2:])
        boxes[:, 2:] = np.maximum(box_source[:, :2], box_source[:, 2:])
        constant = generator.uniform(-10, 10, size=(23))
        min_deviation = generator.uniform(0, 10, size=(31))
        for box in boxes:
            for scalar in constant:
                for multiplier in (-np.inf, np.inf):
                    for min_deviation_row in min_deviation:
                        expected = old_adjust_box_edges(
                            box,
                            (multiplier, scalar),
                            min_deviation=min_deviation_row,
                        )
                        actual = relax.adjust_box_edges(
                            box,
                            ((multiplier, scalar),) * 4,
                            min_deviation=min_deviation_row,
                        )
                        np.testing.assert_array_equal(expected, actual)


if __name__ == "__main__":
    unittest.main()
