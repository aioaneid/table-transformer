import pathlib
import argparse
import xml.etree.ElementTree as ET
import pathlib
import relax
import numpy as np
import sys
import os
import scipy.stats
import collections

sys.path.append("detr")
from util import box_ops

sys.path.append("relax")
import annotations

sys.path.append("src")
import table_datasets


def is_sorted(a):
    return (a[:-1] <= a[1:]).all()


def min_delta(a):
    return np.min(a[1:] - a[:-1]) if a.size >= 2 else 0


def max_delta(a):
    return np.max(a[1:] - a[:-1]) if a.size >= 2 else np.inf


def name_to_path(read_lines):
    # dict maintains iteration order, set does not
    lines = {
        pure_path.name: pure_path
        for pure_path in (pathlib.PurePath(line) for line in read_lines)
    }
    assert len(lines) == len(read_lines)
    return lines


def pascal_voc_metadata(root):
    return (
        root.find("size").find("width").text,
        root.find("size").find("height").text,
        not root.findall("object"),
    )


def cutoff(sorted_paddings, desired_mean_paddings, *, accurate_sum):
    assert is_sorted(sorted_paddings)
    n = len(sorted_paddings)
    if not n:
        return desired_mean_paddings
    spn = sorted_paddings / n
    cs = np.cumsum(spn)
    if accurate_sum:
        for i in range(n):
            x = spn[: i + 1].sum()
            assert np.isclose(cs[i], x)
            cs[i] = x
    assert is_sorted(cs)
    points = cs + spn * np.arange(n - 1, -1, -1)
    min_points_delta = min_delta(points)
    max_points_delta = max_delta(points)
    print("min_points_delta: {}".format(min_points_delta))
    print("max_points_delta: {}".format(max_points_delta))
    if min_points_delta < 0:
        points = np.sort(points)
    right_indices = np.searchsorted(points, desired_mean_paddings, side="right")
    ys = (desired_mean_paddings - cs[np.maximum(right_indices - 1, 0)]) / (
        1 - np.minimum(right_indices, n - 1) / n
    )
    assert right_indices.shape == (len(desired_mean_paddings),)
    assert ys.shape == right_indices.shape

    assert np.logical_or(
        right_indices == 0,
        np.logical_or(
            right_indices == n,
            sorted_paddings[np.minimum(np.maximum(right_indices - 1, 0), n - 1)] <= ys,
        ),
    ).all(), locals()
    assert np.logical_or(
        right_indices == 0,
        np.logical_or(
            right_indices == n, ys < sorted_paddings[np.minimum(right_indices, n - 1)]
        ),
    ).all()
    return np.where(
        right_indices == 0,
        desired_mean_paddings,
        np.where(right_indices == n, np.inf, ys),
    )


def concatenate_list_of_list_of_arrays(values):
    sz = sum([sum([len(a) for a in l], 0) for l in values], 0)
    result = np.empty(sz, dtype=np.float64)
    i = 0
    for l in values:
        for a in l:
            j = i + len(a)
            result[i:j] = a
            i = j
    assert i == sz
    return result
    # return np.concatenate(
    #     sum(values, [np.empty(0, dtype=np.float64)]), dtype=np.float64
    # )


def swap_middle(left_array, right_array):
    t = np.copy(left_array[2:6])
    left_array[2:6] = right_array[2:6]
    right_array[2:6] = t


def convert_label_elements_box_pairs_to_bounds(label_elements_box_triples):
    d = collections.defaultdict(lambda: collections.defaultdict(lambda: [None, None]))
    for label, elements, bbox in label_elements_box_triples:
        m = table_datasets.HOLE_OR_OUTER_LABEL.fullmatch(label)
        assert m
        assert m.group(3) in ["i", "o"]
        outside = m.group(3) == "o"
        io = d[m.group(1)][m.group(2)]
        assert not io[outside]
        io[outside] = (elements, bbox)
    triples = []
    for label, id2io in d.items():
        for _, io in id2io.items():
            assert len(io) == 2
            assert io[0]
            assert io[1]
            triples.append(
                (label, sum([x[0] for x in io], ()), sum([x[1] for x in io], []))
            )
    return triples


def main():
    parser = argparse.ArgumentParser(description="Process tables")
    parser.add_argument(
        "--left_pascal_voc_filelist",
        type=pathlib.PurePath,
        help="Usually px_0 or at least more constrained.",
    )
    parser.add_argument("--left_max_files", type=int, default=2147483647)
    parser.add_argument(
        "--right_pascal_voc_filelist",
        type=pathlib.PurePath,
        help="Usually a (more) relaxed version.",
    )
    parser.add_argument("--right_max_files", type=int, default=2147483647)
    parser.add_argument(
        "--check_non_negative", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--accurate_sum", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--divisions",
        type=int,
        help="1 to not change right due to factor",
        default=10,
    )
    parser.add_argument(
        "--right_output_pascal_voc_xml_dir",
        type=pathlib.PurePath,
        help="Usually a (more) relaxed version.",
    )
    parser.add_argument(
        "--right_margin",
        type=float,
        help="0 to not change right due to margin",
        default=0,
    )
    parser.add_argument(
        "--right_factors",
        type=float,
        help="1 to not change right due to factor",
        default=[1] * 8,
        nargs=8,
    )
    parser.add_argument(
        "--compute_diff",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--print0",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, print a null character after processing each file.",
    )
    args = parser.parse_args()

    left_dir, left_dict = args.left_pascal_voc_filelist.parent, name_to_path(
        relax.read_filelist(args.left_pascal_voc_filelist, args.left_max_files)
    )
    right_dir, right_dict = args.right_pascal_voc_filelist.parent, name_to_path(
        relax.read_filelist(args.right_pascal_voc_filelist, args.right_max_files)
    )
    print("Only left: {}".format(len(left_dict.keys() - right_dict.keys())))
    print("Only right: {}".format(len(right_dict.keys() - left_dict.keys())))
    padding_dict = {}
    for common_key in left_dict.keys() & right_dict.keys():
        left_tree = ET.parse(left_dir.joinpath(left_dict[common_key]))
        right_tree = ET.parse(right_dir.joinpath(right_dict[common_key]))
        assert pascal_voc_metadata(left_tree.getroot()) == pascal_voc_metadata(
            right_tree.getroot()
        ), (
            common_key,
            pascal_voc_metadata(left_tree.getroot()),
            pascal_voc_metadata(right_tree.getroot()),
        )
        left_label_elements_box_triples = (
            annotations.read_label_elements_box_triples_from_root(left_tree.getroot())
        )
        right_label_elements_box_triples = (
            annotations.read_label_elements_box_triples_from_root(right_tree.getroot())
        )
        # Compare e.g. px_0 against pxc_inf. Order of objects should be the same.
        assert [label for label, _, _ in left_label_elements_box_triples] == [
            label for label, _, _ in right_label_elements_box_triples
        ], common_key
        left_label_elements_box_list = convert_label_elements_box_pairs_to_bounds(
            left_label_elements_box_triples
        )
        right_label_elements_box_list = convert_label_elements_box_pairs_to_bounds(
            right_label_elements_box_triples
        )
        assert [label for label, _, _ in left_label_elements_box_list] == [
            label for label, _, _ in right_label_elements_box_list
        ]
        file_diffs = []
        right_factors = np.array(args.right_factors, dtype=np.float64)
        for (_, _, left_box), (_, right_elements, right_box) in zip(
            left_label_elements_box_list, right_label_elements_box_list, strict=True
        ):
            left_array = np.array(left_box, dtype=np.float64)
            right_array = np.array(right_box, dtype=np.float64)
            original_diff = right_array - left_array
            diff = np.copy(original_diff)
            diff[2:6] *= -1
            if args.check_non_negative:
                assert np.all(diff >= 0), (common_key, left_box, right_box, diff)
            if args.compute_diff:
                file_diffs.append(diff)
            if args.right_output_pascal_voc_xml_dir:
                v = np.where(
                    right_factors == 1,
                    right_array,
                    left_array + original_diff * right_factors,
                )
                mx = np.maximum(
                    left_array,
                    np.minimum(
                        v,
                        right_array - args.right_margin,
                    ),
                )
                mn = np.minimum(
                    left_array,
                    np.maximum(
                        v,
                        right_array + args.right_margin,
                    ),
                )
                modified_right_array = np.concatenate((mx[:2], mn[2:6], mx[6:]))
                for right_element, modified_value in zip(
                    right_elements, modified_right_array, strict=True
                ):
                    right_element.text = str(modified_value)
        if args.right_output_pascal_voc_xml_dir:
            output_file_path = os.path.join(
                args.right_output_pascal_voc_xml_dir, common_key
            )
            print(output_file_path)
            right_tree.write(output_file_path)
            if args.print0:
                print("\x00", flush=True)
        padding_dict[common_key] = file_diffs
    all_paddings = concatenate_list_of_list_of_arrays(padding_dict.values())
    if all_paddings.size != 0:
        print(scipy.stats.describe(all_paddings))
        fractions = np.arange(0, 1 + args.divisions) / args.divisions
        avg_padding = np.mean(all_paddings)
        sorted_paddings = np.sort(all_paddings, kind="stable")
        # Not sure why mean of sorted paddings seems to not be better than the mean of all paddings.
        desired_mean_paddings = np.mean(sorted_paddings) * fractions
        paddings = cutoff(
            sorted_paddings,
            desired_mean_paddings,
            accurate_sum=args.accurate_sum,
        )
        for fraction, desired_mean_padding, padding in zip(
            fractions, desired_mean_paddings, paddings
        ):
            print(
                "Cutoff for fraction {} * mean {} = {} padding: {} relaxation: {}".format(
                    fraction, avg_padding, desired_mean_padding, padding, padding * 2
                )
            )


if __name__ == "__main__":
    main()
