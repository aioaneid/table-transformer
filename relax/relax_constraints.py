import pathlib
import argparse
import os
import xml.etree.ElementTree as ET
import relax
import json
import scipy as sp
import numpy as np
import typing
import itertools
import sys
import copy
import datadiff
import datadiff.tools
import hashlib
import collections
import pprint
import time

sys.path.append("src")
sys.path.append("detr")
import eval

TABLE = "table"
TABLE_COLUMN = "table column"
TABLE_ROW = "table row"
TABLE_COLUMN_HEADER = "table column header"
TABLE_PROJECTED_ROW_HEADER = "table projected row header"
TABLE_SPANNING_CELL = "table spanning cell"


def get_args():
    parser = argparse.ArgumentParser(description="Process tables")
    parser.add_argument(
        "--input_pascal_voc_xml_files",
        nargs="*",
        help="Space-separated list of input Pascal POC XML files.",
        type=pathlib.PurePath,
    )
    parser.add_argument(
        "--input_words_data_dir",
        help="Root directory for source data to process",
        type=pathlib.PurePath,
    )
    parser.add_argument(
        "--output_pascal_voc_xml_dir",
        help="Where to output pascal voc XML files.",
        type=pathlib.PurePath,
    )
    parser.add_argument(
        "--centered_hole_side_multiplier",
        type=float,
        help="relax_dataset.inner_multiplier",
        default=1,
    )
    parser.add_argument(
        "--centered_hole_side_constant",
        type=float,
        help="relax_dataset.inner_constant.",
        default=0,
    )
    parser.add_argument(
        "--centered_outer_side_multiplier",
        type=float,
        help="relax_dataset.outer_multiplier",
        default=1,
    )
    parser.add_argument(
        "--centered_outer_side_constant",
        type=float,
        help="relax_dataset.outer_constant",
        default=0,
    )
    parser.add_argument(
        "--table_min_centered_hole_side_constant",
        type=float,
        help="Lower bound on relax_dataset.hole_constant. Usually nonpositive.",
        default=-np.inf,
    )
    parser.add_argument(
        "--table_max_centered_outer_side_constant",
        type=float,
        help="Upper bound on relax_dataset.outer_constant. Usually nonnegative.",
        default=np.inf,
    )
    parser.add_argument(
        "--print0",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, print a null character after processing each file.",
    )
    parser.add_argument(
        "--keep_feasible",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--skip_if_output_exists",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--optimize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=253349686155511324402183545504311281643,
        help="Default obtained with secrets.randbits(128).",
    )
    parser.add_argument(
        "--random_sequential_step",
        type=float,
        nargs="*",
        default=[1],
        help="TODO: Rename to random_sequential_steps. Should be in decreasing order.",
    )
    parser.add_argument(
        "--random_sequential_repeats",
        type=int,
    )
    parser.add_argument("--algo", type=str, default="minimize")
    parser.add_argument("--method", type=str)
    parser.add_argument("--jac", type=str)
    parser.add_argument("--maxiter", type=int)
    parser.add_argument(
        "--include_redundant_token_max_iob_ordering_less_constraints",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--include_unnecessary_token_max_iob_ordering_constraints",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--post_checks",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--post_check_metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--filter_by_iob",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--optimize_linear_constraint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--traverse_nonlinear_constraints_in_lru_order",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--reuse_element_if_one_box",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to reuse the element if outer is identical to inner.",
    )
    parser.add_argument(
        "--dryrun",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to skip writing.",
    )
    parser.add_argument(
        "--stop_file_path",
        help="If this file exists, just wait.",
        type=pathlib.PosixPath,
    )
    parser.add_argument(
        "--stop_file_wait_seconds",
        help="How long to wait before checking again.",
        type=float,
        default=60,
    )
    parser.add_argument("--min_epsilon", type=float, default=0)
    parser.add_argument(
        "--allow_text_constraints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include_token_iob_threshold_constraints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include_token_max_iob_ordering_constraints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include_relative_iob_1d_constraints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include_relative_iob_2d_constraints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--check_original_consistency",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--check_grad",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


# Checks whether iob(interval1, I) >= iob(interva2, I) for any interval I containing x.
def iob_1d_non_increasing(x, interval1, interval2):
    assert is_interval_valid(interval1), interval1
    assert is_interval_valid(interval2), interval2
    a, b = interval1
    c, d = interval2
    if c <= x and x <= d:
        return c <= a and a * x + d * x + b * c == b * x + c * x + a * d
    elif x < c:
        return x <= a and a <= c and b <= d
    else:
        assert d < x
        return d <= b and b <= x and c <= a


# The hole margins are for sure points included in the interval. Any other point in-between as well,
# but we need to go either as much as possible to the left or to the right.
def iob_1d_non_increasing_hole(hole_interval, interval1, interval2):
    return iob_1d_non_increasing(
        hole_interval[0], interval1, interval2
    ) or iob_1d_non_increasing(hole_interval[1], interval1, interval2)


def iob_2d_non_increasing_hole(hole, box1, box2):
    return iob_1d_non_increasing_hole(
        hole[0::2], box1[0::2], box2[0::2]
    ) and iob_1d_non_increasing_hole(hole[1::2], box1[1::2], box2[1::2])


def var_indices_target_box_box_limit_list_for_object_type(
    label_box_list, box_limits, object_type
):
    return [
        (np.repeat(i, repeats=4), target_box, box_limit)
        for i, ((label, target_box), box_limit) in enumerate(
            zip(label_box_list, box_limits, strict=True)
        )
        if label == object_type
    ]


def var_indices_target_box_box_limit_list_of_cell(cell, target_boxes, box_limits):
    return (
        np.array(cell.var_indices()),
        cell.target_box(target_boxes),
        cell.box_limit(box_limits),
    )


class TokenMaxIobOrderingConstraint(typing.NamedTuple):
    token_iob_equals_constraints: list
    token_iob_less_or_equal_constraints: list
    token_iob_less_constraints: list

    def tolist(self):
        return (
            self.token_iob_equals_constraints
            + self.token_iob_less_or_equal_constraints
            + self.token_iob_less_constraints
        )


# Within the maximum iob bucket, the later items in var_indices_target_boxes_box_limits_list have precedence.
# We use this to give precedence to simple cells over supercells.
# It happens sometimes that the text belongs equally well to a simple cell
# and a supercell, with iob < 1. In that case we want to allow the simple cell
# to increase its iob up to and including 1. More often it happens that both the simple
# cell and the supercell have the same iob of 1. In that case we allow the supercell to
# decrease iob up to and including the threshold of 0.001.
def token_max_iob_ordering_constraints(
    token,
    var_indices_target_boxes_box_limits_list,
    token_iob_threshold,
    *,
    n,
    jac,
    keep_feasible,
    min_epsilon: float,
    include_redundant_token_max_iob_ordering_less_constraints,
    include_unnecessary_token_max_iob_ordering_constraints,
):
    # debug_token = (
    #     token
    #     == [341.1321254567511, 69.90323949074002, 373.6631219777753, 81.17181050829902]
    # ).all()
    all_entries = [
        (iob_2d(token, target_box), category, var_indices, target_box, box_limit)
        for category, var_indices_target_boxes_box_limits in enumerate(
            var_indices_target_boxes_box_limits_list
        )
        for var_indices, target_box, box_limit in var_indices_target_boxes_box_limits
    ]
    entries = [
        entry
        for entry in all_entries
        if (
            include_unnecessary_token_max_iob_ordering_constraints
            or entry[0] >= token_iob_threshold
        )
    ]
    if not entries:
        return TokenMaxIobOrderingConstraint(
            token_iob_equals_constraints=[],
            token_iob_less_or_equal_constraints=[],
            token_iob_less_constraints=[],
        )
        # Let the higher category win.
    max_iob, max_category = max(entries, key=lambda entry: entry[:2])[:2]
    max_entries = [entry for entry in entries if entry[:2] == (max_iob, max_category)]
    # if debug_token:
    #     print("max_entries: {}".format(max_entries))
    #     print("iobs: {}".format([entry[:2] for entry in entries]))
    #     print("entries: {}".format(entries))
    if len(max_entries) == 1:
        # Currently one list item has exactly one entry, so we always hit this branch.
        result = TokenMaxIobOrderingConstraint(
            token_iob_equals_constraints=[],
            token_iob_less_or_equal_constraints=[
                token_pair_iob_less_or_equal_constraint(
                    token,
                    var_indices,
                    iob,
                    max_entries[0][2],
                    max_entries[0][0],
                    n=n,
                    jac=jac,
                    keep_feasible=keep_feasible,
                )
                for iob, category, var_indices, _, box_limit in entries
                if category < max_category
                # Reduce number of constraints.
                and (
                    include_redundant_token_max_iob_ordering_less_constraints
                    or intervals_intersect(
                        min_max_iob(token, box_limit),
                        min_max_iob(token, max_entries[0][4]),
                    )
                )
            ],
            token_iob_less_constraints=[
                token_pair_iob_less_constraint(
                    token,
                    var_indices,
                    iob,
                    max_entries[0][2],
                    max_entries[0][0],
                    n=n,
                    jac=jac,
                    keep_feasible=keep_feasible,
                    min_epsilon=min_epsilon,
                )
                for iob, category, var_indices, _, box_limit in entries
                if (
                    (category == max_category and iob != max_iob)
                    or category > max_category
                )
                # Reduce number of constraints.
                and (
                    include_redundant_token_max_iob_ordering_less_constraints
                    or intervals_intersect(
                        min_max_iob(token, box_limit),
                        min_max_iob(token, max_entries[0][4]),
                    )
                )
            ],
        )
        # if debug_token:
        #     print("len(max_entries) = 1 Constraint: {}".format(result))
        return result
    result = TokenMaxIobOrderingConstraint(
        token_iob_equals_constraints=[
            token_iob_equals_constraint(
                token,
                var_indices,
                max_iob,
                n=n,
                jac=jac,
                keep_feasible=keep_feasible,
            )
            for _, _, var_indices, _, _ in max_entries
        ],
        token_iob_less_or_equal_constraints=[
            token_iob_less_or_equal_constraint(
                token,
                var_indices,
                iob,
                max_iob,
                n=n,
                jac=jac,
                keep_feasible=keep_feasible,
            )
            for iob, category, var_indices, _, box_limit in entries
            if category < max_category
            # Reduce number of constraints.
            and (
                include_redundant_token_max_iob_ordering_less_constraints
                or intervals_intersect(
                    min_max_iob(token, box_limit),
                    min_max_iob(token, max_entries[0][4]),
                )
            )
        ],
        token_iob_less_constraints=[
            token_iob_less_constraint(
                token,
                var_indices,
                iob,
                max_iob,
                n=n,
                jac=jac,
                keep_feasible=keep_feasible,
                min_epsilon=min_epsilon,
            )
            for iob, category, var_indices, _, box_limit in entries
            if (
                (category == max_category and iob != max_iob) or category > max_category
            )
            # Reduce number of constraints.
            and (
                include_redundant_token_max_iob_ordering_less_constraints
                or in_interval(max_iob, min_max_iob(token, box_limit))
            )
        ],
    )
    # if debug_token:
    #     print("len(max_entries) > 1 Constraint: {}".format(result))
    return result


def eval_cells(target_boxes, labels, words):
    assert len(target_boxes) == len(labels)
    table_structures, cells, confidence_score = eval.objects_to_cells(
        [target_box.tolist() for target_box in target_boxes],
        [eval.structure_class_map[label] for label in labels],
        [1] * len(labels),
        words,
        # [{'bbox': token.tolist()} for token in tokens],
        eval.structure_class_names,
        eval.structure_class_thresholds,
        eval.structure_class_map,
    )
    # print(pprint.pformat({
    #     'table_structures': table_structures,
    #     'cells': cells,
    #     'confidence_score': confidence_score}))
    return table_structures, cells, confidence_score


def remove_bbox(cells):
    """In the absence of text the bounding boxes are not trimmed to text content so they will be wildly different."""
    result = []
    for cell in cells:
        shallow_copy = {**cell}
        del shallow_copy["bbox"]
        result.append(shallow_copy)
    return result


def wait_if_necessary(stop_file_path, stop_file_wait_seconds):
    if not stop_file_path:
        return
    while stop_file_path.exists():
        print(".", end="", flush=True)
        time.sleep(stop_file_wait_seconds)
    print()


def main():
    args = get_args()
    print({**vars(args), **{"input_pascal_voc_xml_files": None}})

    print("Images: {}".format(len(args.input_pascal_voc_xml_files)), flush=True)
    if args.output_pascal_voc_xml_dir and not args.dryrun:
        os.makedirs(args.output_pascal_voc_xml_dir, exist_ok=True)
    for input_pascal_voc_xml_file in args.input_pascal_voc_xml_files:
        wait_if_necessary(args.stop_file_path, args.stop_file_wait_seconds)
        if args.output_pascal_voc_xml_dir:
            output_file_path = pathlib.PosixPath(
                args.output_pascal_voc_xml_dir, input_pascal_voc_xml_file.name
            )
            if output_file_path.exists() and args.skip_if_output_exists:
                print(
                    "Skipping {} because output exists.".format(
                        input_pascal_voc_xml_file.name
                    )
                )
                continue
        print("Processing: {}".format(input_pascal_voc_xml_file.name))
        start_time_seconds = time.process_time()
        tree = ET.parse(str(input_pascal_voc_xml_file))
        words_filepath = args.input_words_data_dir.joinpath(
            "{}_words.json".format(input_pascal_voc_xml_file.stem)
        )
        with open(words_filepath, "r") as jf:
            words = json.load(jf)
        # words = [word for word in words if word['text'] == 'position']
        # print('words: {}'.format(words))

        default_hole_outer_multiplier_constant = HoleOuterMultiplierConstant(
            hole_side_multiplier=args.centered_hole_side_multiplier,
            hole_side_constant=args.centered_hole_side_constant,
            outer_side_multiplier=args.centered_outer_side_multiplier,
            outer_side_constant=args.centered_outer_side_constant,
        )
        centered_param_dict = dict(
            [
                (label, default_hole_outer_multiplier_constant)
                for label in {
                    TABLE_COLUMN,
                    TABLE_ROW,
                    TABLE_COLUMN_HEADER,
                    TABLE_PROJECTED_ROW_HEADER,
                    TABLE_SPANNING_CELL,
                }
            ]
            + [
                (
                    TABLE,
                    default_hole_outer_multiplier_constant._replace(
                        hole_side_constant=max(
                            default_hole_outer_multiplier_constant.hole_side_constant,
                            args.table_min_centered_hole_side_constant,
                        ),
                        outer_side_constant=min(
                            default_hole_outer_multiplier_constant.outer_side_constant,
                            args.table_max_centered_outer_side_constant,
                        ),
                    ),
                )
            ]
        )
        labels_free_of_text_constraints = {
            TABLE_COLUMN_HEADER,
            TABLE_PROJECTED_ROW_HEADER,
            TABLE_SPANNING_CELL,
        }
        # table_max_centered_outer_side_constant
        for label, hole_outer_multiplier_constant in centered_param_dict.items():
            centered_param_dict[label] = (
                hole_outer_multiplier_constant
                if label in labels_free_of_text_constraints
                or args.allow_text_constraints
                else HoleOuterMultiplierConstant(
                    hole_side_multiplier=1,
                    hole_side_constant=0,
                    outer_side_multiplier=1,
                    outer_side_constant=0,
                )
            )

        success = relax_constraints_in_place(
            tree,
            words,
            centered_param_dict=centered_param_dict,
            optimize=args.optimize,
            algo=args.algo,
            method=args.method,
            file_seed=abs(
                args.seed
                ^ int(
                    hashlib.sha1(
                        input_pascal_voc_xml_file.stem.encode("utf-8")
                    ).hexdigest(),
                    16,
                )
            ),
            maxiter=args.maxiter,
            jac=args.jac,
            keep_feasible=args.keep_feasible,
            random_sequential_step=args.random_sequential_step,
            random_sequential_repeats=args.random_sequential_repeats,
            include_redundant_token_max_iob_ordering_less_constraints=args.include_redundant_token_max_iob_ordering_less_constraints,
            include_unnecessary_token_max_iob_ordering_constraints=args.include_unnecessary_token_max_iob_ordering_constraints,
            post_checks=args.post_checks,
            post_check_metrics=args.post_check_metrics,
            filter_by_iob=args.filter_by_iob,
            optimize_linear_constraint=args.optimize_linear_constraint,
            traverse_nonlinear_constraints_in_lru_order=args.traverse_nonlinear_constraints_in_lru_order,
            reuse_element_if_one_box=args.reuse_element_if_one_box,
            min_epsilon=args.min_epsilon,
            allow_text_constraints=args.allow_text_constraints,
            include_token_iob_threshold_constraints=args.include_token_iob_threshold_constraints,
            include_token_max_iob_ordering_constraints=args.include_token_max_iob_ordering_constraints,
            include_relative_iob_1d_constraints=args.include_relative_iob_1d_constraints,
            include_relative_iob_2d_constraints=args.include_relative_iob_2d_constraints,
            check_original_consistency=args.check_original_consistency,
            check_grad=args.check_grad,
        )
        if not success:
            print("Skipping {} because unsuccessful.".format(input_pascal_voc_xml_file))
            continue
        if args.optimize:
            if args.dryrun:
                if args.output_pascal_voc_xml_dir:
                    print("Not writing in dryrun mode to: {}".format(output_file_path))
            else:
                if args.output_pascal_voc_xml_dir:
                    print(output_file_path)
                    tree.write(output_file_path)
            if args.print0:
                print("\x00", flush=True)
        elapsed_time_seconds = time.process_time() - start_time_seconds
        print("elapsed_time_seconds: {}".format(elapsed_time_seconds))


def interval_length(a):
    assert a[0] <= a[1]
    return a[1] - a[0]


def box_area(box):
    return interval_length(box[::2]) * interval_length(box[1::2])


def is_interval_valid(a):
    return a[0] <= a[1]


class IntervalLimit(typing.NamedTuple):
    hole: np.array
    outer: np.array

    def min_center(self):
        return np.mean([self.outer[0], self.hole[1]])

    def max_center(self):
        return np.mean([self.hole[0], self.outer[1]])

    def min_lower(self):
        return self.outer[0]

    def max_lower(self):
        return self.hole[0]

    def min_upper(self):
        return self.hole[1]

    def max_upper(self):
        return self.outer[1]

    def is_valid_with_absolute_tolerance(self, absolute_tolerance):
        return (
            is_interval_valid(self.hole)
            and is_interval_valid(self.outer)
            and (self.outer[0] <= self.hole[0] + absolute_tolerance).all()
            and (self.hole[1] <= self.outer[1] + absolute_tolerance).all()
        )

    def format(self, interval):
        return (
            "{:6.2f} {} {:6.2f} {} {:6.2f}   {}   {:6.2f} {} {:6.2f} {} {:6.2f}".format(
                *interleave_comp(
                    [
                        self.outer[0],
                        interval[0],
                        self.hole[0],
                        self.hole[1],
                        interval[1],
                        self.outer[1],
                    ]
                )
            )
        )


class BoxLimit(typing.NamedTuple):
    # hole and outer represent only constraints and do not need
    # to represent valid boxes.
    hole: np.array
    outer: np.array

    def x(self):
        return IntervalLimit(hole=interval_x(self.hole), outer=interval_x(self.outer))

    def y(self):
        return IntervalLimit(hole=interval_y(self.hole), outer=interval_y(self.outer))

    def intersect_outer_with_box(self, box):
        return BoxLimit(
            hole=self.hole,
            outer=box_intersection(self.outer, box),
        )

    def enclose_hole_with_limits(self, box):
        return BoxLimit(
            hole=box_enclose(self.hole, box),
            outer=self.outer,
        )

    def intersect(self, box_limit):
        return BoxLimit(
            hole=box_enclose(self.hole, box_limit.hole),
            outer=box_intersection(self.outer, box_limit.outer),
        )

    def is_valid_with_absolute_tolerance(self, absolute_tolerance):
        return self.x().is_valid_with_absolute_tolerance(
            absolute_tolerance
        ) and self.y().is_valid_with_absolute_tolerance(absolute_tolerance)

    def is_valid(self):
        return self.is_valid_with_absolute_tolerance(0)

    def format(self, box):
        return "x: {}         y: {}".format(
            self.x().format(box[::2]), self.y().format(box[1::2])
        )


def box_limit_of_intersection(a, b):
    return BoxLimit(
        hole=box_intersection(a.hole, b.hole), outer=box_intersection(a.outer, b.outer)
    )


def universe_hole():
    return np.array([np.inf, np.inf, -np.inf, -np.inf], dtype=np.float64)


def universe_outer():
    return np.array([-np.inf, -np.inf, np.inf, np.inf], dtype=np.float64)


def universe_box_limit():
    return BoxLimit(hole=universe_hole(), outer=universe_outer())


def post_increment(key, counter):
    count = counter[key]
    counter[key] = count + 1
    return count


def interval_intersection(a, b):
    return max(a[0], b[0]), min(a[1], b[1])


# Does not use length because an empty interval inside another
# interval must be considered as an intersection.
def intervals_intersect(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def signed_interval_length(a):
    return a[1] - a[0]


def unsigned_interval_length(a):
    return np.maximum(signed_interval_length(a), 0)


def unsigned_interval_intersection_length(a, b):
    return unsigned_interval_length(interval_intersection(a, b))


def unsigned_box_area(a):
    return unsigned_interval_length(a[::2]) * unsigned_interval_length(a[1::2])


def signed_box_area(a):
    return signed_interval_length(a[::2]) * signed_interval_length(a[1::2])


def iob_1d(a, b):
    return unsigned_interval_intersection_length(a, b) / (a[1] - a[0])


def iob_2d(a, b):
    return iob_1d(a[0::2], b[0::2]) * iob_1d(a[1::2], b[1::2])


# Approximately 1.5e-5.
strict_inequality_iob_epsilon = 2**-14
nonstrict_inequality_iob_epsilon = 2**-15
epsilon = 2**-18

# Not affected by min_epsilon.
validity_absolute_tolerance = strict_inequality_iob_epsilon
ordering_coordinate_epsilon = strict_inequality_iob_epsilon


def sort_boxes(boxes):
    argsort_tokens = np.vstack(
        [np.argsort(boxes[:, i], kind="stable") for i in range(boxes.shape[-1])]
    )
    sorted_tokens = np.vstack([boxes[row, i] for i, row in enumerate(argsort_tokens)])
    return sorted_tokens, argsort_tokens


def box_intersection(a, b):
    return np.concatenate((np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])))


def box_enclose(a, b):
    return np.concatenate((np.minimum(a[:2], b[:2]), np.maximum(a[2:], b[2:])))


def objective_function(a):
    assert len(a.shape) == 1
    # print('a.shape: {}'.format(a.shape))
    a = a.reshape(len(a) // 8, 2, 4)
    return (-a[:, 0, :2] + a[:, 1, :2] - a[:, 1, 2:] + a[:, 0, 2:]).flatten().mean() / 2


def min_max_iob(token, box_limit):
    assert len(token) == 4
    token_area = signed_box_area(token)
    assert token_area >= 0
    min_iob = unsigned_box_area(box_intersection(token, box_limit.hole)) / token_area
    max_iob = unsigned_box_area(box_intersection(token, box_limit.outer)) / token_area
    return min_iob, max_iob


def different_sides_iob_absolute_box_token(token, box_limit, threshold):
    min_iob, max_iob = min_max_iob(token, box_limit)
    return min_iob < threshold and threshold <= max_iob


def interval_x(box):
    return box[::2]


def interval_y(box):
    return box[1::2]


def min_max_iob_1d_interval_limit_interval(interval_limit, interval2):
    # assert outer[0] <= hole[0] <= hole[1] <= outer[1]
    assert is_interval_valid(interval2)
    left = set(
        [
            interval_limit.outer[0],
            interval_limit.hole[0],
            np.clip(interval2[0], interval_limit.outer[0], interval_limit.hole[0]),
            np.clip(interval2[1], interval_limit.outer[0], interval_limit.hole[0]),
        ]
    )
    right = set(
        [
            interval_limit.hole[1],
            interval_limit.outer[1],
            np.clip(interval2[0], interval_limit.hole[1], interval_limit.outer[1]),
            np.clip(interval2[1], interval_limit.hole[1], interval_limit.outer[1]),
        ]
    )
    iobs = [iob_1d([a, b], interval2) for a in left for b in right]
    return min(iobs), max(iobs)


def min_iob_1d_interval_limit_interval(interval_limit, interval2):
    return min_max_iob_1d_interval_limit_interval(interval_limit, interval2)[0]


def max_iob_1d_interval_limit_interval(interval_limit, interval2):
    return min_max_iob_1d_interval_limit_interval(interval_limit, interval2)[1]


def min_iob_2d_box_limit_box(box_limit, box):
    return (
        min_max_iob_1d_interval_limit_interval(
            box_limit.x(),
            interval_x(box),
        )[0]
        * min_max_iob_1d_interval_limit_interval(
            box_limit.y(),
            interval_y(box),
        )[0]
    )


def max_iob_2d_box_limit_box(box_limit, box):
    return max_iob_1d_interval_limit_interval(
        box_limit.x(),
        interval_x(box),
    ) * max_iob_1d_interval_limit_interval(
        box_limit.y(),
        interval_y(box),
    )


def min_max_iob_2d(box_limit1, box_limit2):
    return min_iob_2d_box_limit_box(
        box_limit1, box_limit2.hole
    ), max_iob_2d_box_limit_box(box_limit1, box_limit2.outer)


def min_max_iob_1d(interval_limit1, interval_limit2):
    return min_iob_1d_interval_limit_interval(
        interval_limit1, interval_limit2.hole
    ), max_iob_1d_interval_limit_interval(interval_limit1, interval_limit2.outer)


def different_sides_absolute_box_absolute_box(box_limit1, box_limit2, threshold):
    assert box_limit1.is_valid_with_absolute_tolerance(validity_absolute_tolerance)
    assert box_limit2.is_valid_with_absolute_tolerance(validity_absolute_tolerance)
    min_iob, max_iob = min_max_iob_2d(box_limit1, box_limit2)
    return min_iob < threshold and threshold <= max_iob


def different_sides_absolute_interval_absolute_interval(
    interval_limit1, interval_limit2, threshold
):
    assert interval_limit1.is_valid_with_absolute_tolerance(validity_absolute_tolerance)
    assert interval_limit2.is_valid_with_absolute_tolerance(validity_absolute_tolerance)
    min_iob, max_iob = min_max_iob_1d(interval_limit1, interval_limit2)
    return min_iob < threshold and threshold <= max_iob


class TracedNonlinearConstrained(typing.NamedTuple):
    flat_var_indices: set
    nonlinear_constraint: sp.optimize.NonlinearConstraint


def token_box_constraint(
    token, target_iob, var_indices, threshold, *, n, jac, keep_feasible, min_epsilon
):
    assert len(var_indices) == 4, var_indices
    left = target_iob < threshold
    lb = (
        0
        if left
        else min(
            target_iob, threshold + max(nonstrict_inequality_iob_epsilon, min_epsilon)
        )
    )
    ub = (
        max(target_iob, threshold - max(strict_inequality_iob_epsilon, min_epsilon))
        if left
        else 1
    )
    is_outer_array = np.repeat(int(left), repeats=4)

    def fun(a):
        assert a.shape == (n << 3,), (a.shape, n)
        a = a.reshape(n, 2, 4)
        # print('a.shape: {}'.format(a.shape))
        extreme_iob = iob_2d(token, a[var_indices, is_outer_array, np.arange(4)])
        return extreme_iob

    return TracedNonlinearConstrained(
        flat_var_indices=set(
            np.ravel_multi_index(
                (var_indices, is_outer_array, np.arange(4)), (n, 2, 4), mode="raise"
            )
        ),
        nonlinear_constraint=sp.optimize.NonlinearConstraint(
            fun, lb, ub, **({"jac": jac} if jac else {}), keep_feasible=keep_feasible
        ),
    )


# Only used if there are no text tokens sufficiently intersecting the table.
def relative_iob_2d_constraint(
    i, target_box1, j, target_box2, threshold, *, n, jac, keep_feasible, min_epsilon
):
    target_box1_area = signed_box_area(target_box1)
    intersection_area = unsigned_box_area(box_intersection(target_box1, target_box2))
    target_iob = intersection_area / target_box1_area
    assert 0 <= target_iob <= 1

    left = target_iob < threshold
    lb = (
        0
        if left
        else min(
            threshold + max(nonstrict_inequality_iob_epsilon, min_epsilon), target_iob
        )
    )
    ub = (
        max(target_iob, threshold - max(strict_inequality_iob_epsilon, min_epsilon))
        if left
        else 1
    )

    def fun(a):
        assert a.shape == (n << 3,), (a.shape, n)
        a = a.reshape(n, 2, 4)
        box_limit1 = BoxLimit(hole=a[i, 0, :], outer=a[i, 1, :])
        assert box_limit1.is_valid_with_absolute_tolerance(epsilon), box_limit1
        box_limit2 = BoxLimit(hole=a[j, 0, :], outer=a[j, 1, :])
        assert box_limit2.is_valid_with_absolute_tolerance(epsilon), box_limit2
        return (
            max_iob_2d_box_limit_box(box_limit1, box_limit2.outer)
            if left
            else min_iob_2d_box_limit_box(box_limit1, box_limit2.hole)
        )

    return TracedNonlinearConstrained(
        flat_var_indices=set(
            np.ravel_multi_index(
                (
                    [i] * 8 + [j] * 4,
                    ([0] * 4 + [1] * 4) + [int(left)] * 4,
                    np.tile(np.arange(4), reps=3),
                ),
                (n, 2, 4),
                mode="raise",
            )
        ),
        nonlinear_constraint=sp.optimize.NonlinearConstraint(
            fun, lb, ub, **({"jac": jac} if jac else {}), keep_feasible=keep_feasible
        ),
    )


def relative_iob_1d_constraint(
    i,
    target_interval1,
    j,
    target_interval2,
    threshold,
    offset: bool,
    *,
    n,
    jac,
    keep_feasible: bool,
    min_epsilon: float,
):
    target_interval1_length = signed_interval_length(target_interval1)
    intersection_length = unsigned_interval_length(
        interval_intersection(target_interval1, target_interval2)
    )
    target_iob = intersection_length / target_interval1_length
    assert 0 <= target_iob <= 1

    left = target_iob < threshold
    lb = (
        0
        if left
        else min(
            threshold + max(nonstrict_inequality_iob_epsilon, min_epsilon), target_iob
        )
    )
    ub = (
        max(target_iob, threshold - max(strict_inequality_iob_epsilon, min_epsilon))
        if left
        else 1
    )

    def fun(a):
        assert a.shape == (n << 3,), (a.shape, n)
        a = a.reshape(n, 2, 4)
        interval_limit1 = IntervalLimit(
            hole=a[i, 0, offset::2], outer=a[i, 1, offset::2]
        )
        assert interval_limit1.is_valid_with_absolute_tolerance(
            epsilon
        ), interval_limit1
        interval_limit2 = IntervalLimit(
            hole=a[j, 0, offset::2], outer=a[j, 1, offset::2]
        )
        assert interval_limit2.is_valid_with_absolute_tolerance(
            epsilon
        ), interval_limit2
        return (
            max_iob_1d_interval_limit_interval(interval_limit1, interval_limit2.outer)
            if left
            else min_iob_1d_interval_limit_interval(
                interval_limit1, interval_limit2.hole
            )
        )

    return TracedNonlinearConstrained(
        flat_var_indices=set(
            np.ravel_multi_index(
                (
                    [i] * 4 + [j] * 2,
                    ([0] * 2 + [1] * 2) + [int(left)] * 2,
                    np.tile(np.arange(offset, 4, 2), reps=3),
                ),
                (n, 2, 4),
                mode="raise",
            )
        ),
        nonlinear_constraint=sp.optimize.NonlinearConstraint(
            fun, lb, ub, **({"jac": jac} if jac else {}), keep_feasible=keep_feasible
        ),
    )


def token_intersection_area(token, a, var_indices, *, is_outer):
    return unsigned_box_area(
        box_intersection(
            token, a[var_indices, np.repeat(int(is_outer), repeats=4), np.arange(4)]
        )
    )


def token_pair_iob_less_constraint(
    token,
    var_indices1,
    target_iob1,
    var_indices2,
    target_iob2,
    *,
    n,
    jac,
    keep_feasible,
    min_epsilon: float,
):
    assert target_iob1 < target_iob2
    token_area = signed_box_area(token)
    assert token_area > 0
    lb, ub = (
        -np.inf,
        max(
            -max(strict_inequality_iob_epsilon, min_epsilon), target_iob1 - target_iob2
        ),
    )
    # print('lb: {} ub: {}'.format(lb, ub))

    def fun(a):
        assert a.shape == (n << 3,), (a.shape, n)
        a = a.reshape(n, 2, 4)

        def max_intersection1():
            return token_intersection_area(token, a, var_indices1, is_outer=True)

        def min_intersection2():
            return token_intersection_area(token, a, var_indices2, is_outer=False)

        # By design the target boxes are within the limits, so
        # if target_iob1 == target_iob2 then the two intervals have at least
        # this target_iob1 * token_area = target_iob2 * token_area in common.
        return max_intersection1() / token_area - min_intersection2() / token_area

    return TracedNonlinearConstrained(
        flat_var_indices=set(
            np.ravel_multi_index(
                (
                    np.concatenate((var_indices1, var_indices2)),
                    [1] * 4 + [0] * 4,
                    np.tile(np.arange(4), reps=2),
                ),
                (n, 2, 4),
                mode="raise",
            )
        ),
        nonlinear_constraint=sp.optimize.NonlinearConstraint(
            fun, lb, ub, **({"jac": jac} if jac else {}), keep_feasible=keep_feasible
        ),
    )


def token_pair_iob_less_or_equal_constraint(
    token,
    var_indices1,
    target_iob1,
    var_indices2,
    target_iob2,
    *,
    n,
    jac,
    keep_feasible,
):
    assert target_iob1 <= target_iob2
    token_area = signed_box_area(token)
    assert token_area > 0
    lb, ub = (-np.inf, 0)
    # print('lb: {} ub: {}'.format(lb, ub))

    def fun(a):
        assert a.shape == (n << 3,), (a.shape, n)
        a = a.reshape(n, 2, 4)

        def max_intersection1():
            return token_intersection_area(token, a, var_indices1, is_outer=True)

        def min_intersection2():
            return token_intersection_area(token, a, var_indices2, is_outer=False)

        # By design the target boxes are within the limits, so
        # if target_iob1 == target_iob2 then the two intervals have at least
        # this target_iob1 * token_area = target_iob2 * token_area in common.
        return max_intersection1() / token_area - min_intersection2() / token_area

    return TracedNonlinearConstrained(
        flat_var_indices=set(
            np.ravel_multi_index(
                (
                    np.concatenate((var_indices1, var_indices2)),
                    [1] * 4 + [0] * 4,
                    np.tile(np.arange(4), reps=2),
                ),
                (n, 2, 4),
                mode="raise",
            )
        ),
        nonlinear_constraint=sp.optimize.NonlinearConstraint(
            fun, lb, ub, **({"jac": jac} if jac else {}), keep_feasible=keep_feasible
        ),
    )


def token_iob_equals_constraint(token, var_indices, iob, *, n, jac, keep_feasible):
    token_area = signed_box_area(token)
    assert token_area > 0
    lb, ub = (iob, iob)

    def fun(a):
        assert a.shape == (n << 3,), (a.shape, n)
        a = a.reshape(n, 2, 4)
        return np.array(
            [
                token_intersection_area(token, a, var_indices, is_outer=False)
                / token_area,
                token_intersection_area(token, a, var_indices, is_outer=True)
                / token_area,
            ],
            dtype=np.float64,
        )

    return TracedNonlinearConstrained(
        flat_var_indices=set(
            np.ravel_multi_index(
                (
                    var_indices + var_indices,
                    [0] * 4 + [1] * 4,
                    np.tile(np.arange(4), reps=2),
                ),
                (n, 2, 4),
                mode="raise",
            )
        ),
        nonlinear_constraint=sp.optimize.NonlinearConstraint(
            fun, lb, ub, **({"jac": jac} if jac else {}), keep_feasible=keep_feasible
        ),
    )


def token_iob_less_constraint(
    token,
    var_indices,
    iob,
    max_iob,
    *,
    n,
    jac,
    keep_feasible,
    min_epsilon: float,
):
    token_area = signed_box_area(token)
    assert token_area > 0
    assert iob < max_iob
    lb, ub = (
        -np.inf,
        max(max_iob - max(strict_inequality_iob_epsilon, min_epsilon), iob),
    )

    def fun(a):
        assert a.shape == (n << 3,), (a.shape, n)
        a = a.reshape(n, 2, 4)
        return (
            token_intersection_area(token, a, var_indices, is_outer=True) / token_area
        )

    return TracedNonlinearConstrained(
        flat_var_indices=set(
            np.ravel_multi_index(
                (var_indices, [1] * 4, np.arange(4)), (n, 2, 4), mode="raise"
            )
        ),
        nonlinear_constraint=sp.optimize.NonlinearConstraint(
            fun, lb, ub, **({"jac": jac} if jac else {}), keep_feasible=keep_feasible
        ),
    )


def token_iob_less_or_equal_constraint(
    token, var_indices, iob, max_iob, *, n, jac, keep_feasible
):
    token_area = signed_box_area(token)
    assert token_area > 0
    assert iob <= max_iob
    lb, ub = (-np.inf, max_iob)

    def fun(a):
        assert a.shape == (n << 3,), (a.shape, n)
        a = a.reshape(n, 2, 4)
        return (
            token_intersection_area(token, a, var_indices, is_outer=True) / token_area
        )

    return TracedNonlinearConstrained(
        flat_var_indices=set(
            np.ravel_multi_index(
                (var_indices, [1] * 4, np.arange(4)), (n, 2, 4), mode="raise"
            )
        ),
        nonlinear_constraint=sp.optimize.NonlinearConstraint(
            fun, lb, ub, **({"jac": jac} if jac else {}), keep_feasible=keep_feasible
        ),
    )


def in_interval(x, interval):
    return interval[0] <= x <= interval[1]


def zeros_with_ones(n, ones):
    a = np.zeros((n,), dtype=np.float64)
    a[ones] = 1
    return a


def zeros_with_ones_and_minus_ones(n, ones, minus_ones):
    a = zeros_with_ones(n, ones)
    a[minus_ones] = -1
    return a


def comp(a, b):
    return "<" if a < b else "=" if a == b else ">"


def interleave_comp(a):
    return sum([[a[i], comp(a[i], a[i + 1])] for i in range(len(a) - 1)], []) + a[-1:]


def relative_iob_2d_constraints_for_object_types(
    label_left,
    label_right,
    iob_threshold,
    label_box_list,
    box_limits,
    *,
    jac,
    keep_feasible,
    min_epsilon: float,
):
    return tuple(
        relative_iob_2d_constraint(
            i,
            target_box1,
            j,
            target_box2,
            iob_threshold,
            n=len(box_limits),
            jac=jac,
            keep_feasible=keep_feasible,
            min_epsilon=min_epsilon,
        )
        for i, ((label1, target_box1), box_limit1) in enumerate(
            zip(label_box_list, box_limits)
        )
        if label1 == label_left
        for j, ((label2, target_box2), box_limit2) in enumerate(
            zip(label_box_list, box_limits)
        )
        if label2 == label_right and i != j  # For the common case when labels coincide.
        # Reduce number of constraints.
        and different_sides_absolute_box_absolute_box(
            box_limit1,
            box_limit2,
            iob_threshold,
        )
    )


def relative_iob_1d_constraints_for_object_types(
    label_left,
    label_right_set,
    iob_threshold,
    label_interval_list,
    interval_limits,
    offset: bool,
    *,
    jac,
    keep_feasible,
    min_epsilon: float,
):
    assert len(label_interval_list) == len(interval_limits)
    return tuple(
        relative_iob_1d_constraint(
            i,
            target_interval1,
            j,
            target_interval2,
            iob_threshold,
            offset,
            n=len(label_interval_list),
            jac=jac,
            keep_feasible=keep_feasible,
            min_epsilon=min_epsilon,
        )
        for i, ((label1, target_interval1), interval_limit1) in enumerate(
            zip(label_interval_list, interval_limits)
        )
        if label1 == label_left
        for j, ((label2, target_interval2), interval_limit2) in enumerate(
            zip(label_interval_list, interval_limits)
        )
        if label2 in label_right_set
        and i != j  # For the never happening case when labels coincide.
        # Reduce number of constraints.
        and different_sides_absolute_interval_absolute_interval(
            interval_limit1,
            interval_limit2,
            iob_threshold,
        )
    )


def row_col_ordering_params(
    label_box_list,
    box_limits,
    *,
    key_function,
    min_offsets,
    max_offsets,
    factor,
    lower_bound_function,
    upper_bound_function,
):
    n = len(label_box_list)
    assert len(box_limits) == n
    params = []
    for object_type in (TABLE_COLUMN, TABLE_ROW):
        index = object_type == TABLE_ROW
        entries = sorted(
            [
                (
                    key_function((interval_x, interval_y)[index](target_box)),
                    i,
                    target_box,
                    (box_limit.x(), box_limit.y())[index],
                )
                for i, ((label, target_box), box_limit) in enumerate(
                    zip(label_box_list, box_limits, strict=True)
                )
                if label == object_type
            ]
        )
        sorted_groups = [
            (k, list(g)) for k, g in itertools.groupby(entries, lambda entry: entry[0])
        ]
        # print('sorted_groups: {}'.format(sorted_groups))
        for k, g in sorted_groups:
            if len(g) == 1:
                continue
            # hole must be identical to outer, and both have the given center
            params.extend(
                sum(
                    [
                        [
                            (
                                k,
                                zeros_with_ones(
                                    8 * n, [i * 8 + index + x for x in min_offsets]
                                )
                                * factor,
                                k,
                            ),
                            (
                                k,
                                zeros_with_ones(
                                    8 * n, [i * 8 + index + x for x in max_offsets]
                                )
                                * factor,
                                k,
                            ),
                        ]
                        for _, i, _, _ in g
                    ],
                    [],
                )
            )
        params.extend(
            [
                (
                    -np.inf,
                    zeros_with_ones_and_minus_ones(
                        8 * n,
                        [g1[0][1] * 8 + index + x for x in max_offsets],
                        [g2[0][1] * 8 + index + x for x in min_offsets],
                    )
                    * factor,
                    max(-ordering_coordinate_epsilon, k1 - k2),
                )
                for (k1, g1), (k2, g2) in itertools.pairwise(sorted_groups)
                if len(g1) == len(g2) == 1
                # Reduce the number of constraints
                and intervals_intersect(
                    (lower_bound_function(g1[0][3]), upper_bound_function(g1[0][3])),
                    (lower_bound_function(g2[0][3]), upper_bound_function(g2[0][3])),
                )
            ]
        )
        params.extend(
            [
                (
                    -np.inf,
                    zeros_with_ones(
                        8 * n,
                        [g1[0][1] * 8 + index + x for x in max_offsets],
                    )
                    * factor,
                    max(k2 - ordering_coordinate_epsilon, k1),
                )
                for (k1, g1), (k2, g2) in itertools.pairwise(sorted_groups)
                if len(g1) == 1 and len(g2) != 1
                # Reduce the number of constraints
                and in_interval(
                    k2, (lower_bound_function(g1[0][3]), upper_bound_function(g1[0][3]))
                )
            ]
        )
        params.extend(
            [
                (
                    min(k1 + ordering_coordinate_epsilon, k2),
                    zeros_with_ones(
                        8 * n, [g2[0][1] * 8 + index + x for x in min_offsets]
                    )
                    * factor,
                    np.inf,
                )
                for (k1, g1), (k2, g2) in itertools.pairwise(sorted_groups)
                if len(g1) != 1 and len(g2) == 1
                # Reduce the number of constraints
                and in_interval(
                    k1, (lower_bound_function(g2[0][3]), upper_bound_function(g2[0][3]))
                )
            ]
        )
    return params


def compute_row_col_center_params(label_box_list, box_limits):
    return row_col_ordering_params(
        label_box_list,
        box_limits,
        key_function=lambda interval: interval.mean(),
        min_offsets=(2, 4),
        max_offsets=(0, 6),
        factor=0.5,
        lower_bound_function=lambda interval_limit: interval_limit.min_center(),
        upper_bound_function=lambda interval_limit: interval_limit.max_center(),
    )


def compute_row_col_lower_params(label_box_list, box_limits):
    return row_col_ordering_params(
        label_box_list,
        box_limits,
        key_function=lambda interval: interval[0],
        min_offsets=(4,),
        max_offsets=(0,),
        factor=1,
        lower_bound_function=lambda interval_limit: interval_limit.min_lower(),
        upper_bound_function=lambda interval_limit: interval_limit.max_lower(),
    )


def compute_row_col_upper_params(label_box_list, box_limits):
    return row_col_ordering_params(
        label_box_list,
        box_limits,
        key_function=lambda interval: interval[1],
        min_offsets=(2,),
        max_offsets=(6,),
        factor=1,
        lower_bound_function=lambda interval_limit: interval_limit.min_upper(),
        upper_bound_function=lambda interval_limit: interval_limit.max_upper(),
    )


def linear_constraint_from_params(linear_params, *, var_count, keep_feasible):
    return sp.optimize.LinearConstraint(
        A=np.array([params[1] for params in linear_params], dtype=np.float64).reshape(
            -1, var_count
        ),
        lb=np.array([params[0] for params in linear_params], dtype=np.float64),
        ub=np.array([params[2] for params in linear_params], dtype=np.float64),
        keep_feasible=keep_feasible,
    )


def compute_different_sides_tokens(tokens, target_box, box_limit, threshold):
    return [
        (token, iob_2d(token, target_box) >= threshold)
        for token in tokens
        # Reduce number of constraints.
        if different_sides_iob_absolute_box_token(token, box_limit, threshold)
    ]


def filtered_tokens_for_token_iob_threshold_constraints(
    different_sides_tokens, box_limit
):
    # print("different_sides_tokens: {}".format(different_sides_tokens))

    # print("len(different_sides_tokens): {}".format(len(different_sides_tokens)))
    # Further reduce the number of constraints.
    filtered_tokens = [
        token
        for i, (token, iob_ge_threshold) in enumerate(different_sides_tokens)
        if not any(
            (
                iob_ge_threshold == igt
                and (
                    iob_2d_non_increasing_hole(box_limit.hole, token, tk)
                    if igt
                    else iob_2d_non_increasing_hole(box_limit.hole, tk, token)
                )
                and (
                    i < j
                    or not (
                        iob_2d_non_increasing_hole(box_limit.hole, tk, token)
                        if igt
                        else iob_2d_non_increasing_hole(box_limit.hole, token, tk)
                    )
                )
                for j, (tk, igt) in enumerate(different_sides_tokens)
            )
        )
    ]
    # print("len(filtered_tokens): {}".format(len(filtered_tokens)))
    # print("filtered_tokens: {}".format(filtered_tokens))
    return filtered_tokens


def token_iob_threshold_constraints(
    tokens,
    indices_target_box_box_limit_list,
    threshold,
    *,
    n,
    jac,
    keep_feasible,
    filter_by_iob,
    min_epsilon: float,
):
    return tuple(
        token_box_constraint(
            token,
            iob_2d(token, target_box),
            indices,
            threshold,
            n=n,
            jac=jac,
            keep_feasible=keep_feasible,
            min_epsilon=min_epsilon,
        )
        for indices, target_box, box_limit in indices_target_box_box_limit_list
        for token in (
            filtered_tokens_for_token_iob_threshold_constraints(
                compute_different_sides_tokens(
                    tokens, target_box, box_limit, threshold
                ),
                box_limit,
            )
            if filter_by_iob
            else [
                tk
                for tk, _ in compute_different_sides_tokens(
                    tokens, target_box, box_limit, threshold
                )
            ]
        )
    )


def violated_bounds(bounds, x, allowed_violation):
    sl, sb = bounds.residual(x)
    return ((sl < -allowed_violation) | (sb < -allowed_violation)).sum()


def linear_constraint_violation(linear_constraint, x_flat, allowed_violation):
    sl, sb = linear_constraint.residual(x_flat)
    if (sl < -allowed_violation).any() or (sb < -allowed_violation).any():
        return {"sl": sl, "sb": sb}
    return None


def nonlinear_constraint_violation(nonlinear_constraint, x_flat, allowed_violation):
    objective = nonlinear_constraint.fun(x_flat)
    if (nonlinear_constraint.lb - allowed_violation > objective).any() or (
        objective > nonlinear_constraint.ub + allowed_violation
    ).any():
        return {
            "lb": nonlinear_constraint.lb,
            "objective": objective,
            "ub": nonlinear_constraint.ub,
        }
    return None


def lru_find_and_update_first_violated_nonlinear_constraint(
    x_flat,
    allowed_violation,
    traverse_nonlinear_constraints_in_lru_order,
):
    for (
        key,
        nonlinear_constraint,
    ) in traverse_nonlinear_constraints_in_lru_order.items():
        violation = nonlinear_constraint_violation(
            nonlinear_constraint, x_flat, allowed_violation
        )
        if violation:
            traverse_nonlinear_constraints_in_lru_order.move_to_end(key, last=False)
            return violation
    return None


def find_first_violated_nonlinear_constraint(
    nonlinear_constraints, x_flat, allowed_violation
):
    for i, nonlinear_constraint in enumerate(nonlinear_constraints):
        violation = nonlinear_constraint_violation(
            nonlinear_constraint, x_flat, allowed_violation
        )
        if violation:
            # print(
            #     "index: {} step: {} Violation index: {} / {}".format(
            #         index, step, i, len(nonlinear_constraints)
            #     )
            # )
            return violation
    # print("index: {} step: {} okay: {}".format(index, step, len(nonlinear_constraints)))
    return None


def violated_constraints(
    linear_constraint, nonlinear_constraints, x_flat, allowed_violation
):
    result = []
    if linear_constraint:
        violation = linear_constraint_violation(
            linear_constraint, x_flat, allowed_violation
        )
        if violation:
            result.append(violation)
    for nonlinear_constraint in nonlinear_constraints:
        violation = nonlinear_constraint_violation(
            nonlinear_constraint, x_flat, allowed_violation
        )
        if violation:
            result.append(violation)
    return result


def filter_linear_constraint_by_flat_var_index(linear_constraint, var_index):
    indicator = linear_constraint.A[:, var_index].astype(bool)
    return sp.optimize.LinearConstraint(
        A=linear_constraint.A[indicator],
        lb=linear_constraint.lb[indicator],
        ub=linear_constraint.ub[indicator],
    )


def random_sequential(
    objective_function,
    linear_constraint,
    traced_nonlinear_constraints,
    *,
    x0,
    bounds: sp.optimize.Bounds,
    max_steps,
    seed,
    maxiter,
    random_sequential_repeats,
    optimize_linear_constraint,
    traverse_nonlinear_constraints_in_lru_order,
):
    random_generator = np.random.default_rng(seed)
    x = x0.copy()
    y = x.copy()
    nit = 0
    if optimize_linear_constraint:
        linear_constraint_by_flat_var_index = [
            filter_linear_constraint_by_flat_var_index(linear_constraint, i)
            for i in range(len(x0))
        ]
        # for i, lc in enumerate(linear_constraint_by_flat_var_index):
        #     print(i, lc.A.shape)
    nonlinear_constraint_indices_by_flat_var_index = [set() for i in range(len(x0))]
    for i, traced_nonlinear_constraint in enumerate(traced_nonlinear_constraints):
        # print('i: {} flat_var_indices: {}'.format(i, traced_nonlinear_constraint.flat_var_indices))
        for flat_var_index in traced_nonlinear_constraint.flat_var_indices:
            nonlinear_constraint_indices_by_flat_var_index[flat_var_index].add(i)

    nonlinear_constraints_by_flat_var_index = [
        [
            traced_nonlinear_constraints[x].nonlinear_constraint
            for x in nonlinear_constraint_indices
        ]
        for nonlinear_constraint_indices in nonlinear_constraint_indices_by_flat_var_index
    ]
    if traverse_nonlinear_constraints_in_lru_order:
        traverse_nonlinear_constraints_in_lru_order_by_flat_var_index = [
            # The key is irrelevant, it just needs to be unique.
            # It is only required because the standard library does
            # not have an OrderedList class.
            collections.OrderedDict(enumerate(nonlinear_constraints))
            for nonlinear_constraints in nonlinear_constraints_by_flat_var_index
        ]
        del nonlinear_constraint_indices_by_flat_var_index
    # print(
    #     "nonlinear_constraint_indices_by_flat_var_index: {}".format(
    #         nonlinear_constraint_indices_by_flat_var_index
    #     )
    # )
    r = np.arange(len(x0))
    allowed_violation = 0
    for max_step in max_steps:
        # print("max_step: {}".format(max_step))
        while maxiter is None or nit < maxiter:
            indices = np.concatenate(
                [random_generator.permuted(r) for _ in range(random_sequential_repeats)]
            )
            steps = (
                random_generator.random((len(x0) * random_sequential_repeats,))
                * max_step
            )
            # print("indices: {} steps: {}".format(indices, steps))
            for index, step in zip(indices, steps, strict=True):
                outer = (index >> 2) & 1
                increase = ((index & 3) >= 2) == outer
                old_value = y[index]
                y[index] = old_value + (1 if increase else -1) * step
                if (
                    violated_bounds(bounds, y, allowed_violation)
                    or linear_constraint_violation(
                        (
                            linear_constraint_by_flat_var_index[index]
                            if optimize_linear_constraint
                            else linear_constraint
                        ),
                        y,
                        allowed_violation,
                    )
                    or (
                        lru_find_and_update_first_violated_nonlinear_constraint(
                            y,
                            allowed_violation,
                            traverse_nonlinear_constraints_in_lru_order_by_flat_var_index[
                                index
                            ],
                        )
                        if traverse_nonlinear_constraints_in_lru_order
                        else find_first_violated_nonlinear_constraint(
                            nonlinear_constraints_by_flat_var_index[index],
                            y,
                            allowed_violation,
                        )
                    )
                ):
                    y[index] = old_value
            nit += random_sequential_repeats
            if (x == y).all():
                break
            np.copyto(x, y)
    return sp.optimize.OptimizeResult(
        {
            "x": x,
            "success": True,
            "fun": objective_function(x),
            "nit": nit,
        }
    )


def indexed_label_box_list(label_box_list, label_set):
    return [
        (i, label, box)
        for (i, (label, box)) in enumerate(label_box_list)
        if label in label_set
    ]


def target_box_of_var_indices(var_indices, target_boxes):
    return np.array(
        [target_boxes[var_index][i] for i, var_index in enumerate(var_indices)],
        dtype=np.float64,
    )


def box_limit_of_var_indices(var_indices, box_limits):
    return BoxLimit(
        hole=np.array(
            [box_limits[var_index].hole[i] for i, var_index in enumerate(var_indices)],
            dtype=np.float64,
        ),
        outer=np.array(
            [box_limits[var_index].outer[i] for i, var_index in enumerate(var_indices)],
            dtype=np.float64,
        ),
    )


class SupercellPoint(typing.NamedTuple):
    row_index: int
    col_index: int


class Supercell(typing.NamedTuple):
    # inclusive
    lower: SupercellPoint
    # inclusive
    upper: SupercellPoint

    def row_count(self):
        return self.upper.row_index - self.lower.row_index + 1

    def col_count(self):
        return self.upper.col_index - self.lower.col_index + 1

    def var_indices(self):
        """x comes from min/max column, y comes from min/max row"""
        return [
            self.lower.col_index,
            self.lower.row_index,
            self.upper.col_index,
            self.upper.row_index,
        ]

    def target_box(self, target_boxes):
        return target_box_of_var_indices(self.var_indices(), target_boxes)

    def box_limit(self, box_limits):
        return box_limit_of_var_indices(self.var_indices(), box_limits)


class SimpleCell(typing.NamedTuple):
    row_index: int
    col_index: int

    def var_indices(self):
        """x comes from min/max column, y comes from min/max row"""
        return [self.col_index, self.row_index, self.col_index, self.row_index]

    def target_box(self, target_boxes):
        return target_box_of_var_indices(self.var_indices(), target_boxes)

    def box_limit(self, box_limits):
        return box_limit_of_var_indices(self.var_indices(), box_limits)


def extract_supercells(
    label_box_list, *, iob_row_supercell_y_threshold, iob_col_supercell_x_threshold
):
    result = []
    for supercell_index, supercell_label, supercell_box in indexed_label_box_list(
        label_box_list, {TABLE_PROJECTED_ROW_HEADER, TABLE_SPANNING_CELL}
    ):
        y_offset = True
        matching_row_indices = [
            j
            for (j, _, row_box) in indexed_label_box_list(label_box_list, {TABLE_ROW})
            if iob_1d(row_box[y_offset::2], supercell_box[y_offset::2])
            >= iob_row_supercell_y_threshold
        ]
        assert matching_row_indices, (supercell_index, supercell_label, supercell_box)
        assert (
            min(matching_row_indices) == matching_row_indices[0]
        ), matching_row_indices
        assert (
            max(matching_row_indices) == matching_row_indices[-1]
        ), matching_row_indices
        # TODO: Stable-sort the input rows e.g. by vertical center before making these assertions.
        # The function table_structure_to_cells in slot_into_containers also does that, but *after*
        # calling slot_into_containers, so ties within that function will depend on the input order.
        # This looks like a bug which needs to be fixed.
        for i in range(matching_row_indices[0] + 1, matching_row_indices[-1]):
            if label_box_list[i][0] == TABLE_ROW:
                assert i in matching_row_indices, (
                    "The input boxes should have been sorted so the matching rows are contiguous.",
                    i,
                    matching_row_indices,
                )
        for index, j in enumerate(matching_row_indices):
            if label_box_list[j][1][1] < label_box_list[matching_row_indices[0]][1][1]:
                print(
                    "Input rows (at least the intersecting ones) should have been sorted by min_y: {}".format(
                        matching_row_indices
                    )
                )
                matching_row_indices[0], matching_row_indices[index] = (
                    matching_row_indices[index],
                    matching_row_indices[0],
                )
        for index, j in enumerate(matching_row_indices):
            if label_box_list[j][1][3] > label_box_list[matching_row_indices[-1]][1][3]:
                print(
                    "Input rows (at least the intersecting ones) should have been sorted by max_y: {}".format(
                        matching_row_indices
                    )
                )
                matching_row_indices[-1], matching_row_indices[index] = (
                    matching_row_indices[index],
                    matching_row_indices[-1],
                )
        x_offset = False
        matching_col_indices = [
            j
            for (j, _, col_box) in indexed_label_box_list(
                label_box_list, {TABLE_COLUMN}
            )
            if iob_1d(col_box[x_offset::2], supercell_box[x_offset::2])
            >= iob_col_supercell_x_threshold
        ]
        assert matching_col_indices, (supercell_index, supercell_label, supercell_box)

        # assert (
        #     min(matching_col_indices) == matching_col_indices[0]
        # ), matching_col_indices
        # assert (
        #     max(matching_col_indices) == matching_col_indices[-1]
        # ), matching_col_indices

        # TODO: Sort the input columns e.g. by horizontal center before making these assertions.
        for i in range(matching_col_indices[0] + 1, matching_col_indices[-1]):
            if label_box_list[i][0] == TABLE_COLUMN:
                assert i in matching_col_indices, (
                    "The input boxes should have been sorted so the matching columns are contiguous.",
                    i,
                    matching_col_indices,
                )
        for index, j in enumerate(matching_col_indices):
            if label_box_list[j][1][0] < label_box_list[matching_col_indices[0]][1][0]:
                print(
                    "Input cols (at least the intersecting ones) should have been sorted by min_x: {}".format(
                        matching_col_indices
                    )
                )
                matching_col_indices[0], matching_col_indices[index] = (
                    matching_col_indices[index],
                    matching_col_indices[0],
                )
        for index, j in enumerate(matching_col_indices):
            if label_box_list[j][1][2] > label_box_list[matching_col_indices[-1]][1][2]:
                print(
                    "Input cols (at least the intersecting ones) should have been sorted by max_x: {}".format(
                        matching_col_indices
                    )
                )
                matching_col_indices[-1], matching_col_indices[index] = (
                    matching_col_indices[index],
                    matching_col_indices[-1],
                )
        result.append(
            Supercell(
                lower=SupercellPoint(
                    row_index=matching_row_indices[0],
                    col_index=matching_col_indices[0],
                ),
                upper=SupercellPoint(
                    row_index=matching_row_indices[-1],
                    col_index=matching_col_indices[-1],
                ),
            )
        )

    return result


def extract_simple_cells(label_box_list):
    # The order must correspond to the one in table_structure_to_cells.
    return [
        SimpleCell(row_index=row_index, col_index=col_index)
        for col_index, _, _ in indexed_label_box_list(label_box_list, {TABLE_COLUMN})
        for row_index, _, _ in indexed_label_box_list(label_box_list, {TABLE_ROW})
    ]


class HoleOuterMultiplierConstant(typing.NamedTuple):
    hole_side_multiplier: float
    hole_side_constant: float
    outer_side_multiplier: float
    outer_side_constant: float


def count_differences(original_table_structure_counts, counter):
    pairs = {
        "rows": {TABLE_ROW},
        "columns": {TABLE_COLUMN},
        "headers": {TABLE_COLUMN_HEADER},
        "supercells": {TABLE_PROJECTED_ROW_HEADER, TABLE_SPANNING_CELL},
    }
    return {
        key
        for key, value_set in pairs.items()
        if original_table_structure_counts[key]
        != sum([counter[value] for value in value_set])
    }


def relax_constraints_in_place(
    tree,
    words,
    centered_param_dict,
    *,
    optimize,
    algo,
    method,
    file_seed,
    maxiter,
    jac,
    keep_feasible,
    random_sequential_step,
    random_sequential_repeats,
    include_redundant_token_max_iob_ordering_less_constraints,
    include_unnecessary_token_max_iob_ordering_constraints,
    post_checks,
    post_check_metrics,
    filter_by_iob,
    optimize_linear_constraint,
    traverse_nonlinear_constraints_in_lru_order,
    reuse_element_if_one_box,
    min_epsilon: float,
    allow_text_constraints: bool,
    include_token_iob_threshold_constraints: bool,
    include_token_max_iob_ordering_constraints: bool,
    include_relative_iob_1d_constraints: bool,
    include_relative_iob_2d_constraints: bool,
    check_original_consistency: bool,
    check_grad: bool,
):
    print("file_seed: {}".format(file_seed))
    # print("jac: {}".format(repr(jac)))
    root = tree.getroot()
    size_element = root.find("size")
    width = float(size_element.find("width").text)
    height = float(size_element.find("height").text)

    objects = root.findall("object")

    obj_label_box_list = []

    for obj in objects:
        obj_label_box_list.append(
            (
                obj,
                obj.find("name").text,
                np.array(relax.read_bbox_of_object(obj), dtype=np.float64),
            )
        )
    label_box_list = [(label, box) for _, label, box in obj_label_box_list]
    for box1, box2 in itertools.pairwise([box for (label, box) in label_box_list if label == TABLE_ROW]):
        if box1[1] + box1[3] > box2[1] + box2[3]:
            print('Rows in unexpected order: {} {}'.format(box1, box2))
    for box1, box2 in itertools.pairwise([box for (label, box) in label_box_list if label == TABLE_COLUMN]):
        if box1[0] + box1[2] > box2[0] + box2[2]:
            print('Columns in unexpected order: {} {}'.format(box1, box2))

    n = len(label_box_list)
    assert len([_ for label, _ in label_box_list if label == TABLE]) == 1

    target_boxes = np.array([box for _, box in label_box_list], dtype=np.float64)

    assert (0 <= target_boxes[:, 0:2]).all(), target_boxes
    assert (target_boxes[:, 0:2] <= target_boxes[:, 2:4]).all(), target_boxes
    assert (target_boxes[:, 2:4] <= np.array([width, height])).all(), (
        target_boxes,
        width,
        height,
    )
    labels = [label for label, _ in label_box_list]

    if optimize or check_original_consistency:
        # print("tokens: {}".format(tokens))
        original_table_structures, original_cells, _ = eval_cells(
            target_boxes, labels, words
        )
        counter = collections.Counter([label for label, _ in label_box_list])
        original_table_structure_counts = {
            key: len(values) for key, values in original_table_structures.items()
        }
        pprint.pprint(
            {
                "original_table_structure_counts": original_table_structure_counts,
                "counter": counter,
            }
        )
        if cd := count_differences(original_table_structure_counts, counter):
            print(
                "Different counts detected: {} between original_table_structure_counts: {} and counter: {}".format(
                    cd, original_table_structure_counts, counter
                )
            )
            return False
        # if (counter[TABLE_ROW] and counter[TABLE_COLUMN] > 1) or True:

        original_table_structures_no_words, original_cells_no_words, _ = eval_cells(
            target_boxes, labels, []
        )
        # print(original_cells_no_words)

    text_boxes = tuple(word["bbox"] for word in words)
    tokens = np.array(text_boxes, dtype=np.float64).reshape(-1, 4)
    assert (tokens[:, :2] <= tokens[:, 2:]).all(), tokens

    # print("table_structures: {}".format(table_structures))
    # print("original_cells: {}".format(original_cells))
    # print("confidence_score: {}".format(confidence_score))

    # To ensure order is preserved amongst the variables it would
    # be enough to ensure o^u_1 <= o^l_2 because of
    # u_1 <= o^u_1 and o^l_2 <= l_2. But explicit hard constraints
    # help us get better 1d inequalities when transforming 2D iob
    # inequalities into 1D.
    image_box = np.array(
        [
            nonstrict_inequality_iob_epsilon,
            nonstrict_inequality_iob_epsilon,
            width - nonstrict_inequality_iob_epsilon,
            height - nonstrict_inequality_iob_epsilon,
        ],
        dtype=np.float64,
    )
    intersecting_hole_borders = [
        relax.adjust_box_edges(
            target_box,
            (
                (
                    centered_param_dict[label].hole_side_multiplier,
                    centered_param_dict[label].hole_side_constant,
                ),
            )
            * 4,
            # Make the hole area positive.
            min_deviation=nonstrict_inequality_iob_epsilon,
        )
        for target_box, label in zip(target_boxes, labels)
    ]
    intersecting_outer_borders = [
        box_intersection(
            relax.adjust_box_edges(
                target_box,
                (
                    (
                        centered_param_dict[label].outer_side_multiplier,
                        centered_param_dict[label].outer_side_constant,
                    ),
                )
                * 4,
                min_deviation=0,
            ),
            image_box,
        )
        for target_box, label in zip(target_boxes, labels)
    ]
    box_limits = [
        universe_box_limit().intersect(
            BoxLimit(hole=intersecting_hole_border, outer=intersecting_outer_border)
        )
        for intersecting_hole_border, intersecting_outer_border in zip(
            intersecting_hole_borders, intersecting_outer_borders
        )
    ]
    print(
        "Objective without constraints: {}".format(
            objective_function(
                np.array(
                    [[box_limit.hole, box_limit.outer] for box_limit in box_limits]
                ).flatten()
            )
        )
    )

    for (
        box_limit,
        target_box,
        label,
        intersecting_hole_border,
        intersecting_outer_border,
    ) in zip(
        box_limits,
        target_boxes,
        labels,
        intersecting_hole_borders,
        intersecting_outer_borders,
    ):
        assert box_limit.is_valid(), pprint.pformat(
            {
                "box_limit": box_limit,
                "target_box": target_box,
                "label": label,
                "hole_side_multiplier": centered_param_dict[label].hole_side_multiplier,
                "hole_side_constant": centered_param_dict[label].hole_side_constant,
                "intersecting_hole_border": intersecting_hole_border,
                "intersecting_outer_border": intersecting_outer_border,
            }
        )

    # Solution: [[hole, outer] for i in range(n)]
    x0 = np.tile(target_boxes[:, None, :], [1, 2, 1])
    # print("x0: {}".format(x0))
    lb = np.array(
        [
            [
                # min hole coordinates
                np.concatenate([target_box[:2], box_limit.hole[2:]]),
                # min outer coordinates
                np.concatenate([box_limit.outer[:2], target_box[2:]]),
            ]
            for (target_box, box_limit) in zip(target_boxes, box_limits, strict=True)
        ],
        dtype=np.float64,
    )
    ub = np.array(
        [
            [
                # max hole coordinates
                np.concatenate([box_limit.hole[:2], target_box[2:]]),
                # max outer coordinates
                np.concatenate([target_box[:2], box_limit.outer[2:]]),
            ]
            for (target_box, box_limit) in zip(target_boxes, box_limits, strict=True)
        ],
        dtype=np.float64,
    )

    token_table_iob_threshold = 0.5
    token_indices = [
        i
        for i, token in enumerate(tokens)
        if [
            target_box
            for (label, target_box) in label_box_list
            if label == TABLE and iob_2d(token, target_box) >= token_table_iob_threshold
        ]
    ]
    token_table_iob_constraints = (
        token_iob_threshold_constraints(
            tokens,
            [
                (np.repeat(i, 4), box, box_limit)
                for i, ((label, box), box_limit) in enumerate(
                    zip(label_box_list, box_limits, strict=True)
                )
                if label == TABLE
            ],
            token_table_iob_threshold,
            n=n,
            jac=jac,
            keep_feasible=keep_feasible,
            filter_by_iob=filter_by_iob,
            min_epsilon=min_epsilon,
        )
        if include_token_iob_threshold_constraints
        else ()
    )
    print("token_table_iob_constraints: {}".format(len(token_table_iob_constraints)))

    relevant_tokens = [tokens[index] for index in token_indices]
    # print(
    #     "relevant_tokens: {} all_tokens: {}".format(len(relevant_tokens), len(tokens))
    # )

    del tokens

    # A supercell is considered to span min top-y to max bottom-y of its rows and similarly for its
    # columns. In order to keep the same row and column numbers having the min/max, and for aesthetic
    # reasons, let's ensure the row y endpoints and the column x endpoints remain in the same order.
    # This is in addition to the ordering of their centers. The latter would become redundant if we
    # knew for sure that in the training data there are no rows included in other rows. Let's keep
    # those constraints as well, just in case.

    row_row_iob_constraints = (
        relative_iob_2d_constraints_for_object_types(
            TABLE_ROW,
            TABLE_ROW,
            0.5,
            label_box_list,
            box_limits,
            jac=jac,
            keep_feasible=keep_feasible,
            min_epsilon=min_epsilon,
        )
        if include_relative_iob_2d_constraints
        else ()
    )
    # if not relevant_tokens or True
    # else ()
    print("row_row_iob_constraints: {}".format(len(row_row_iob_constraints)))
    col_col_iob_constraints = (
        relative_iob_2d_constraints_for_object_types(
            TABLE_COLUMN,
            TABLE_COLUMN,
            0.25,
            label_box_list,
            box_limits,
            jac=jac,
            keep_feasible=keep_feasible,
            min_epsilon=min_epsilon,
        )
        if include_relative_iob_2d_constraints
        else ()
    )
    # if not relevant_tokens or True
    # else ()
    print("col_col_iob_constraints: {}".format(len(col_col_iob_constraints)))

    row_var_indices_target_boxes_box_limits = (
        var_indices_target_box_box_limit_list_for_object_type(
            label_box_list, box_limits, TABLE_ROW
        )
    )

    token_iob_threshold = 0.5

    token_row_iob_threshold = token_iob_threshold
    token_row_iob_constraints = (
        token_iob_threshold_constraints(
            relevant_tokens,
            row_var_indices_target_boxes_box_limits,
            token_row_iob_threshold,
            n=n,
            jac=jac,
            keep_feasible=keep_feasible,
            filter_by_iob=filter_by_iob,
            min_epsilon=min_epsilon,
        )
        if include_token_iob_threshold_constraints
        else ()
    )
    print("token_row_iob_constraints: {}".format(len(token_row_iob_constraints)))
    token_row_max_iob_ordering_constraints = tuple(
        sum(
            [
                (
                    token_max_iob_ordering_constraints(
                        token,
                        reversed(
                            [[x] for x in row_var_indices_target_boxes_box_limits]
                        ),
                        token_row_iob_threshold,
                        n=n,
                        jac=jac,
                        keep_feasible=keep_feasible,
                        min_epsilon=min_epsilon,
                        include_redundant_token_max_iob_ordering_less_constraints=include_redundant_token_max_iob_ordering_less_constraints,
                        include_unnecessary_token_max_iob_ordering_constraints=include_unnecessary_token_max_iob_ordering_constraints,
                    )
                    if include_token_max_iob_ordering_constraints
                    else TokenMaxIobOrderingConstraint(
                        token_iob_equals_constraints=[],
                        token_iob_less_or_equal_constraints=[],
                        token_iob_less_constraints=[],
                    )
                ).tolist()
                for token in relevant_tokens
            ],
            [],
        )
    )
    print(
        "token_row_max_iob_ordering_constraints: {}".format(
            len(token_row_max_iob_ordering_constraints)
        )
    )

    col_var_indices_target_boxes_box_limits = (
        var_indices_target_box_box_limit_list_for_object_type(
            label_box_list, box_limits, TABLE_COLUMN
        )
    )
    token_column_iob_threshold = token_iob_threshold
    token_column_iob_constraints = (
        token_iob_threshold_constraints(
            relevant_tokens,
            col_var_indices_target_boxes_box_limits,
            token_column_iob_threshold,
            n=n,
            jac=jac,
            keep_feasible=keep_feasible,
            filter_by_iob=filter_by_iob,
            min_epsilon=min_epsilon,
        )
        if include_token_iob_threshold_constraints
        else ()
    )
    print("token_column_iob_constraints: {}".format(len(token_column_iob_constraints)))
    token_col_max_iob_ordering_constraints = tuple(
        sum(
            [
                (
                    token_max_iob_ordering_constraints(
                        token,
                        reversed(
                            [[x] for x in col_var_indices_target_boxes_box_limits]
                        ),
                        token_column_iob_threshold,
                        n=n,
                        jac=jac,
                        keep_feasible=keep_feasible,
                        min_epsilon=min_epsilon,
                        include_redundant_token_max_iob_ordering_less_constraints=include_redundant_token_max_iob_ordering_less_constraints,
                        include_unnecessary_token_max_iob_ordering_constraints=include_unnecessary_token_max_iob_ordering_constraints,
                    )
                    if include_token_max_iob_ordering_constraints
                    else TokenMaxIobOrderingConstraint(
                        token_iob_equals_constraints=[],
                        token_iob_less_or_equal_constraints=[],
                        token_iob_less_constraints=[],
                    )
                ).tolist()
                for token in relevant_tokens
            ],
            [],
        )
    )
    print(
        "token_col_max_iob_ordering_constraints: {}".format(
            len(token_col_max_iob_ordering_constraints)
        )
    )

    row_col_lower_params = compute_row_col_lower_params(label_box_list, box_limits)
    print("  row_col_lower_params: {}".format(len(row_col_lower_params)))

    row_col_center_params = compute_row_col_center_params(label_box_list, box_limits)
    print("  row_col_center_params: {}".format(len(row_col_center_params)))

    row_col_upper_params = compute_row_col_upper_params(label_box_list, box_limits)
    print("  row_col_upper_params: {}".format(len(row_col_upper_params)))

    linear_params = (
        [] + row_col_lower_params + row_col_center_params + row_col_upper_params
    )

    linear_constraint = linear_constraint_from_params(
        linear_params, var_count=n * 8, keep_feasible=keep_feasible
    )
    print(
        "row_col_ordering_constraints (linear): {}".format(
            len(linear_constraint.lb) if linear_constraint else 0
        )
    )

    y_offset = True
    # This check happens once before refine_rows, and then once again
    # if there are at least 1 row and 2 columns. In the latter case however
    # we could remove some constraints by stopping at the second row (sorted by y-mean) which is not part of the header,
    # if header starts at a row > 0; otherwise only up to and including the first such row, and
    # by considering that for each row, we
    # only really need to preserve the condition "exists header s.t. iob(row, header) >= 0.5".
    iob_row_header_y_constraints = (
        relative_iob_1d_constraints_for_object_types(
            TABLE_ROW,
            {TABLE_COLUMN_HEADER},
            0.5,
            [(label, box[y_offset::2]) for label, box in label_box_list],
            [box_limit.y() for box_limit in box_limits],
            y_offset,
            jac=jac,
            keep_feasible=keep_feasible,
            min_epsilon=min_epsilon,
        )
        if include_relative_iob_1d_constraints
        else ()
    )
    print("iob_row_header_y_constraints: {}".format(len(iob_row_header_y_constraints)))

    iob_row_supercell_y_threshold = 0.5
    iob_col_supercell_x_threshold = 0.5
    # Actually all the ground trugh tables seem to have a single header, so this is a no-op.
    iob_header_header_constraints = (
        relative_iob_2d_constraints_for_object_types(
            TABLE_COLUMN_HEADER,
            TABLE_COLUMN_HEADER,
            0.05,
            label_box_list,
            box_limits,
            jac=jac,
            keep_feasible=keep_feasible,
            min_epsilon=min_epsilon,
        )
        if include_relative_iob_2d_constraints
        else ()
    )
    print(
        "iob_header_header_constraints: {}".format(len(iob_header_header_constraints))
    )
    row_supercell_y_iob_1d_constraints = (
        relative_iob_1d_constraints_for_object_types(
            TABLE_ROW,
            {TABLE_PROJECTED_ROW_HEADER, TABLE_SPANNING_CELL},
            iob_row_supercell_y_threshold,
            [(label, box[y_offset::2]) for label, box in label_box_list],
            [box_limit.y() for box_limit in box_limits],
            y_offset,
            jac=jac,
            keep_feasible=keep_feasible,
            min_epsilon=min_epsilon,
        )
        if include_relative_iob_1d_constraints
        else ()
    )
    print(
        "row_supercell_y_iob_1d_constraints: {}".format(
            len(row_supercell_y_iob_1d_constraints)
        )
    )
    x_offset = False
    col_supercell_x_iob_1d_constraints = (
        relative_iob_1d_constraints_for_object_types(
            TABLE_COLUMN,
            {TABLE_PROJECTED_ROW_HEADER, TABLE_SPANNING_CELL},
            iob_col_supercell_x_threshold,
            [(label, box[x_offset::2]) for label, box in label_box_list],
            [box_limit.x() for box_limit in box_limits],
            x_offset,
            jac=jac,
            keep_feasible=keep_feasible,
            min_epsilon=min_epsilon,
        )
        if include_relative_iob_1d_constraints
        else ()
    )
    print(
        "col_supercell_x_iob_1d_constraints: {}".format(
            len(col_supercell_x_iob_1d_constraints)
        )
    )
    # end if

    supercells = extract_supercells(
        label_box_list,
        iob_row_supercell_y_threshold=iob_row_supercell_y_threshold,
        iob_col_supercell_x_threshold=iob_col_supercell_x_threshold,
    )
    simple_cells = extract_simple_cells(label_box_list)

    token_cell_iob_threshold = 0.001

    simple_cell_var_indices_target_boxes_box_limits = [
        var_indices_target_box_box_limit_list_of_cell(cell, target_boxes, box_limits)
        for cell in simple_cells
    ]
    print(
        "simple_cell_var_indices_target_boxes_box_limits: {}".format(
            len(simple_cell_var_indices_target_boxes_box_limits)
        )
    )
    token_simple_cell_iob_constraints = (
        token_iob_threshold_constraints(
            relevant_tokens,
            simple_cell_var_indices_target_boxes_box_limits,
            token_cell_iob_threshold,
            n=n,
            jac=jac,
            keep_feasible=keep_feasible,
            filter_by_iob=filter_by_iob,
            min_epsilon=min_epsilon,
        )
        if include_token_iob_threshold_constraints
        else ()
    )
    print(
        "token_simple_cell_iob_constraints: {}".format(
            len(token_simple_cell_iob_constraints)
        )
    )

    supercell_var_indices_target_boxes_box_limits = [
        var_indices_target_box_box_limit_list_of_cell(cell, target_boxes, box_limits)
        for cell in supercells
    ]
    print(
        "supercell_var_indices_target_boxes_box_limits: {}".format(
            len(supercell_var_indices_target_boxes_box_limits)
        )
    )
    token_supercell_iob_constraints = (
        token_iob_threshold_constraints(
            relevant_tokens,
            supercell_var_indices_target_boxes_box_limits,
            token_cell_iob_threshold,
            n=n,
            jac=jac,
            keep_feasible=keep_feasible,
            filter_by_iob=filter_by_iob,
            min_epsilon=min_epsilon,
        )
        if include_token_iob_threshold_constraints
        else ()
    )
    print(
        "token_supercell_iob_constraints: {}".format(
            len(token_supercell_iob_constraints)
        )
    )

    token_cell_max_iob_ordering_constraints = [
        (
            token_max_iob_ordering_constraints(
                token,
                reversed(
                    [[x] for x in simple_cell_var_indices_target_boxes_box_limits]
                    + [[x] for x in supercell_var_indices_target_boxes_box_limits]
                ),
                token_cell_iob_threshold,
                n=n,
                jac=jac,
                keep_feasible=keep_feasible,
                min_epsilon=min_epsilon,
                include_redundant_token_max_iob_ordering_less_constraints=include_redundant_token_max_iob_ordering_less_constraints,
                include_unnecessary_token_max_iob_ordering_constraints=include_unnecessary_token_max_iob_ordering_constraints,
            )
            if include_token_max_iob_ordering_constraints
            else TokenMaxIobOrderingConstraint(
                token_iob_equals_constraints=[],
                token_iob_less_or_equal_constraints=[],
                token_iob_less_constraints=[],
            )
        )
        for token in relevant_tokens
    ]

    token_cell_iob_equals_constraints = tuple(
        sum(
            [
                x.token_iob_equals_constraints
                for x in token_cell_max_iob_ordering_constraints
            ],
            [],
        )
    )
    print(
        "token_cell_iob_equals_constraints: {}".format(
            len(token_cell_iob_equals_constraints)
        )
    )
    token_cell_iob_less_or_equal_constraints = tuple(
        sum(
            [
                x.token_iob_less_or_equal_constraints
                for x in token_cell_max_iob_ordering_constraints
            ],
            [],
        )
    )
    print(
        "token_cell_iob_less_or_equal_constraints: {}".format(
            len(token_cell_iob_less_or_equal_constraints)
        )
    )
    token_cell_iob_less_constraints = tuple(
        sum(
            [
                x.token_iob_less_constraints
                for x in token_cell_max_iob_ordering_constraints
            ],
            [],
        )
    )
    print(
        "token_cell_iob_less_constraints: {}".format(
            len(token_cell_iob_less_constraints)
        )
    )

    # Assume that the supercells in the training data are already
    # consistent, e.g. no overlaps, if in header starting then each
    # all supercells from row 0 up to start of supercell, tree
    # supercell in header etc.

    all_traced_nonlinear_constraints = (
        ()
        + token_table_iob_constraints
        + token_row_iob_constraints
        + token_column_iob_constraints
        + row_row_iob_constraints
        + col_col_iob_constraints
        + token_row_max_iob_ordering_constraints
        + token_col_max_iob_ordering_constraints
        + iob_row_header_y_constraints
        + iob_header_header_constraints
        + row_supercell_y_iob_1d_constraints
        + col_supercell_x_iob_1d_constraints
        + token_simple_cell_iob_constraints
        + token_supercell_iob_constraints
        + token_cell_iob_equals_constraints
        + token_cell_iob_less_or_equal_constraints
        + token_cell_iob_less_constraints
    )
    traced_nonlinear_constraints = (
        all_traced_nonlinear_constraints
        if allow_text_constraints
        else (
            ()
            + iob_row_header_y_constraints
            + iob_header_header_constraints
            + row_supercell_y_iob_1d_constraints
            + col_supercell_x_iob_1d_constraints
        )
    )
    check_nonlinear_constraints = tuple(
        [x.nonlinear_constraint for x in all_traced_nonlinear_constraints]
    )
    check_linear_constraint = linear_constraint
    train_linear_constraint = (
        linear_constraint
        if allow_text_constraints
        else linear_constraint_from_params(
            [], var_count=n * 8, keep_feasible=keep_feasible
        )
    )
    all_train_constraints = tuple(
        [x.nonlinear_constraint for x in traced_nonlinear_constraints]
    ) + (train_linear_constraint,)

    bounds = sp.optimize.Bounds(
        lb=lb.flatten(),
        ub=ub.flatten(),
        keep_feasible=keep_feasible,
    )
    z = np.zeros((8 * n,), dtype=np.float64)
    zz = np.zeros((8 * n, 8 * n), dtype=np.float64)
    jacobian = np.tile(
        (np.array([[-1, -1, 1, 1], [1, 1, -1, -1]]) / (4 * n))[None, ...], (n, 1, 1)
    ).flatten()

    if check_grad:
        assert (
            sp.optimize.check_grad(
                objective_function,
                lambda x: jacobian,
                x0=np.random.default_rng(file_seed).random((8 * n,)) * 100,
            )
            < epsilon
        )
    x0_flat = x0.flatten()
    allowed_violation = (
        min(strict_inequality_iob_epsilon, nonstrict_inequality_iob_epsilon) / 2
    )
    # Allowed violation should be at the beginning zero, except for numerical errors.
    assert not violated_bounds(bounds, x0_flat, allowed_violation=allowed_violation**2)
    assert not violated_constraints(
        check_linear_constraint,
        check_nonlinear_constraints,
        x0_flat,
        allowed_violation=allowed_violation**2,
    )
    if not optimize:
        print("Not optimizing, returning.")
        return True
    match algo:
        case "minimize":
            result = sp.optimize.minimize(
                objective_function,
                x0=x0_flat,
                bounds=bounds,
                constraints=all_train_constraints,
                jac=lambda x: jacobian,
                hess=lambda x: zz,
                hessp=lambda x, p: z,
                method=method,
                options={
                    **({"maxiter": maxiter} if maxiter else {}),
                    "disp": True,
                    **({"iprint": 2} if method == "SLSQP" else {}),
                },
            )
        case "shgo":
            result = sp.optimize.shgo(
                objective_function,
                bounds=bounds,
                constraints=all_train_constraints,
                options={
                    "jac": lambda x: jacobian,
                    "hess": lambda x: zz,
                    "hessp": lambda x, p: z,
                    "disp": True,
                },
            )
        case "differential_evolution":
            result = sp.optimize.differential_evolution(
                objective_function,
                x0=x0_flat,
                bounds=bounds,
                constraints=all_train_constraints,
                seed=file_seed,
                disp=True,
            )
        case "random_sequential":
            result = random_sequential(
                objective_function,
                train_linear_constraint,
                traced_nonlinear_constraints,
                x0=x0_flat,
                bounds=bounds,
                max_steps=random_sequential_step,
                seed=file_seed,
                maxiter=maxiter,
                random_sequential_repeats=random_sequential_repeats,
                optimize_linear_constraint=optimize_linear_constraint,
                traverse_nonlinear_constraints_in_lru_order=traverse_nonlinear_constraints_in_lru_order,
            )
        case "random_sequential_then_minimize":
            random_seq_result = random_sequential(
                objective_function,
                train_linear_constraint,
                traced_nonlinear_constraints,
                x0=x0_flat,
                bounds=bounds,
                max_steps=random_sequential_step,
                seed=file_seed,
                maxiter=maxiter,
                random_sequential_repeats=random_sequential_repeats,
                optimize_linear_constraint=optimize_linear_constraint,
                traverse_nonlinear_constraints_in_lru_order=traverse_nonlinear_constraints_in_lru_order,
            )
            print("random_seq_result: {}".format(random_seq_result))
            assert not violated_constraints(
                check_linear_constraint,
                check_nonlinear_constraints,
                random_seq_result.x,
                allowed_violation,
            ), violated_constraints(
                check_linear_constraint,
                check_nonlinear_constraints,
                random_seq_result.x,
                allowed_violation,
            )
            minimize_result = sp.optimize.minimize(
                objective_function,
                x0=random_seq_result.x,
                bounds=bounds,
                constraints=all_train_constraints,
                jac=lambda x: jacobian,
                hess=lambda x: zz,
                hessp=lambda x, p: z,
                method=method,
                options={
                    **({"maxiter": maxiter} if maxiter else {}),
                    "disp": True,
                    **({"iprint": 2} if method == "SLSQP" else {}),
                },
            )
            print("minimize_result: {}".format(minimize_result))
            result = minimize_result if minimize_result.success else random_seq_result

    print("result: {}".format(result))
    if not result.success:
        result = random_sequential(
            objective_function,
            train_linear_constraint,
            traced_nonlinear_constraints,
            x0=x0_flat,
            bounds=bounds,
            max_steps=random_sequential_step,
            seed=file_seed,
            maxiter=maxiter,
            random_sequential_repeats=random_sequential_repeats,
            optimize_linear_constraint=optimize_linear_constraint,
            traverse_nonlinear_constraints_in_lru_order=traverse_nonlinear_constraints_in_lru_order,
        )
    assert result.success
    assert result.fun <= 0
    optimal_box_limits = [
        BoxLimit(hole=optimal_relaxed_box[0], outer=optimal_relaxed_box[1])
        for optimal_relaxed_box in result.x.reshape(-1, 2, 4)
    ]

    if post_checks:
        assert not violated_constraints(
            check_linear_constraint,
            check_nonlinear_constraints,
            result.x,
            allowed_violation,
        ), violated_constraints(
            check_linear_constraint,
            check_nonlinear_constraints,
            result.x,
            allowed_violation,
        )
        for (_, label, target_box), optimal_box_limit in zip(
            obj_label_box_list, optimal_box_limits, strict=True
        ):
            print(
                "{:4} {}".format(
                    "".join(w[0] for w in label.split(" ")),
                    optimal_box_limit.format(target_box),
                )
            )

        original_cells_without_bbox = remove_bbox(original_cells)
        original_cells_no_words_without_bbox = remove_bbox(original_cells_no_words)

        hole_table_structures_no_words, hole_cells_no_words, _ = eval_cells(
            [optimal_box_limit.hole for optimal_box_limit in optimal_box_limits],
            labels,
            [],
        )
        # print("hole_cells: {}".format(hole_cells))
        datadiff.tools.assert_equal(
            original_cells_no_words_without_bbox, remove_bbox(hole_cells_no_words)
        )
        # pprint.pprint(
        #     {
        #         "hole": {
        #             "table_structures": hole_table_structures_no_words,
        #             "cells": hole_cells_no_words,
        #         },
        #         "original": {
        #             "table_structures": original_table_structures_no_words,
        #             "cells": original_cells_no_words,
        #         },
        #     }
        # )
        outer_table_structures_no_words, outer_cells_no_words, _ = eval_cells(
            [optimal_box_limit.outer for optimal_box_limit in optimal_box_limits],
            labels,
            [],
        )
        # print("outer_cells: {}".format(outer_cells))
        datadiff.tools.assert_equal(
            original_cells_no_words_without_bbox, remove_bbox(outer_cells_no_words)
        )
        # pprint.pprint(
        #     {
        #         "outer": {
        #             "table_structures": outer_table_structures_no_words,
        #             "cells": outer_cells_no_words,
        #         },
        #         "original": {
        #             "table_structures": original_table_structures_no_words,
        #             "cells": original_cells_no_words,
        #         },
        #     }
        # )

        hole_table_structures, hole_cells, _ = eval_cells(
            [optimal_box_limit.hole for optimal_box_limit in optimal_box_limits],
            labels,
            words,
        )
        # print("hole_cells: {}".format(hole_cells))
        datadiff.tools.assert_equal(
            original_cells_without_bbox, remove_bbox(hole_cells)
        )
        if post_check_metrics:
            hole_metrics = eval.compute_metrics(
                "grits",
                target_boxes,
                labels,
                [1] * len(labels),
                original_cells,
                [optimal_box_limit.hole for optimal_box_limit in optimal_box_limits],
                labels,
                [1] * len(labels),
                hole_cells,
                debug_grits=False,
            )
            # Example with grits_con != 1: PMC4698666_table_3.xml
            # hole_metrics: {'grits_top': 1.0, 'grits_precision_top': 1.0, 'grits_recall_top': 1.0, 'grits_top_upper_bound': 1.0, 'grits_loc': 0.9982852192116686, 'grits_precision_loc': 0.9982852192116686, 'grits_recall_loc': 0.9982852192116686, 'grits_loc_upper_bound': 0.9982852192116686, 'grits_con': 1.0, 'grits_precision_con': 1.0, 'grits_recall_con': 1.0, 'grits_con_upper_bound': 1.0, 'acc_con': 1}
            print("hole_metrics: {}".format(hole_metrics))

        outer_table_structures, outer_cells, _ = eval_cells(
            [optimal_box_limit.outer for optimal_box_limit in optimal_box_limits],
            labels,
            words,
        )
        # print("outer_cells: {}".format(outer_cells))
        datadiff.tools.assert_equal(
            original_cells_without_bbox, remove_bbox(outer_cells)
        )
        if post_check_metrics:
            outer_metrics = eval.compute_metrics(
                "grits",
                target_boxes,
                labels,
                [1] * len(labels),
                original_cells,
                [optimal_box_limit.outer for optimal_box_limit in optimal_box_limits],
                labels,
                [1] * len(labels),
                outer_cells,
                debug_grits=False,
            )
            # Example with grits_con != 1: PMC4698666_table_3.xml
            # outer_metrics: {'grits_top': 1.0, 'grits_precision_top': 1.0, 'grits_recall_top': 1.0, 'grits_top_upper_bound': 1.0, 'grits_loc': 0.9967071851363256, 'grits_precision_loc': 0.9967071851363256, 'grits_recall_loc': 0.9967071851363256, 'grits_loc_upper_bound': 0.9967071851363256, 'grits_con': 1.0, 'grits_precision_con': 1.0, 'grits_recall_con': 1.0, 'grits_con_upper_bound': 1.0, 'acc_con': 1}
            print("outer_metrics: {}".format(outer_metrics))

    for tag, (obj, optimal_box_limit) in enumerate(zip(objects, optimal_box_limits)):
        name_element = obj.find("name")
        name = name_element.text

        box_element = obj.find("bndbox")
        x_min_element = box_element.find("xmin")
        y_min_element = box_element.find("ymin")
        x_max_element = box_element.find("xmax")
        y_max_element = box_element.find("ymax")
        obj2 = copy.deepcopy(obj)

        (
            x_min_element.text,
            y_min_element.text,
            x_max_element.text,
            y_max_element.text,
        ) = map(str, optimal_box_limit.hole)
        if (
            np.array_equal(optimal_box_limit.hole, optimal_box_limit.outer)
            and reuse_element_if_one_box
        ):
            continue
        name_element.text = "{} {} i".format(name, tag)
        obj2.find("name").text = "{} {} o".format(name, tag)
        box_element2 = obj2.find("bndbox")
        (
            box_element2.find("xmin").text,
            box_element2.find("ymin").text,
            box_element2.find("xmax").text,
            box_element2.find("ymax").text,
        ) = map(str, optimal_box_limit.outer)
        root.append(obj2)

    return True


if __name__ == "__main__":
    main()
