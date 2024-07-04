import copy
from enum import Enum
import math
import itertools
import annotations


class OuterClamp(Enum):
    NONE = 0
    IMAGE = 1
    TABLE = 2
    ORIGINAL_OR_TABLE = 3


class InnerEnclose(Enum):
    NONE = 0
    ORIGINAL_NEAR_TABLE_EDGES = 1


def adjust_corner(corner, *, factor, constant, min_deviation):
    if math.isinf(factor):
        return (
            -max(min_deviation / 2, corner[1] / 2 - corner[0] / 2 + constant / 2)
            * factor
        )
    comp_func = min if factor >= 0 else max
    return comp_func(
        corner[0] / 2 + corner[1] / 2 - min_deviation / 2 * factor,
        corner[0] * (0.5 + factor / 2)
        + corner[1] * (0.5 - factor / 2)
        - constant * (factor / 2),
    )


def adjust_box_edges(box, multiplier_and_constant_list, *, min_deviation):
    assert len(multiplier_and_constant_list) == 4, multiplier_and_constant_list
    for multiplier_and_constant in multiplier_and_constant_list:
        assert len(multiplier_and_constant) == 2, multiplier_and_constant
    return (
        adjust_corner(
            box[0::2],
            factor=multiplier_and_constant_list[0][0],
            constant=multiplier_and_constant_list[0][1],
            min_deviation=min_deviation,
        ),
        adjust_corner(
            box[1::2],
            factor=multiplier_and_constant_list[1][0],
            constant=multiplier_and_constant_list[1][1],
            min_deviation=min_deviation,
        ),
        adjust_corner(
            box[0::2],
            factor=-multiplier_and_constant_list[2][0],
            constant=multiplier_and_constant_list[2][1],
            min_deviation=min_deviation,
        ),
        adjust_corner(
            box[1::2],
            factor=-multiplier_and_constant_list[3][0],
            constant=multiplier_and_constant_list[3][1],
            min_deviation=min_deviation,
        ),
    )


def clamp_to_bounding_box(box, bounding_box):
    return tuple((max if i < 2 else min)(box[i], bounding_box[i]) for i in range(4))


def enclose_bounding_box(box, bounding_box):
    return tuple((min if i < 2 else max)(box[i], bounding_box[i]) for i in range(4))


def read_bbox_of_object(obj):
    box_element = obj.find("bndbox")
    x_min_element = box_element.find("xmin")
    y_min_element = box_element.find("ymin")
    x_max_element = box_element.find("xmax")
    y_max_element = box_element.find("ymax")

    x_min = float(x_min_element.text)
    y_min = float(y_min_element.text)
    x_max = float(x_max_element.text)
    y_max = float(y_max_element.text)

    # Equality allowed because relax-detection-random-px.sh creates
    # such cases, e.g. PMC6289453_5.
    assert x_min <= x_max
    assert y_min <= y_max
    return x_min, y_min, x_max, y_max


# Usually left <= original <= right
def make_symmetric_1d(original, left, right):
    d = min(original - left, right - original)
    return (left if d == original - left else original - d), (
        right if d == right - original else original + d
    )


def modify_object_in_place(
    root,
    tag,
    obj,
    outer_multiplier_and_constant_list,
    outer_boundary_clamp_box,
    inner_multiplier_and_constant_list,
    inner_boundary_enclose_box,
    clamp_inner_boundary_to_outer_boundary,
    swap_inner_boundary_corners,
    reuse_element_if_one_box,
    *,
    symmetric_hole_and_outer: bool
):
    name_element = obj.find("name")
    name = name_element.text

    x_min_element, y_min_element, x_max_element, y_max_element = (
        annotations.get_bounding_box_elements(obj)
    )
    original_box = annotations.get_bounding_box_of_elements(
        x_min_element, y_min_element, x_max_element, y_max_element
    )

    outer_boundary = adjust_box_edges(
        original_box,
        outer_multiplier_and_constant_list,
        min_deviation=0,
    )
    clamped_outer_boundary = clamp_to_bounding_box(
        outer_boundary,
        outer_boundary_clamp_box,
    )
    del outer_boundary
    # print("final_outer_boundary: {}".format(final_outer_boundary))
    inner_boundary = adjust_box_edges(
        original_box,
        inner_multiplier_and_constant_list,
        min_deviation=0,
    )
    expanded_inner_boundary = enclose_bounding_box(
        inner_boundary,
        inner_boundary_enclose_box,
    )
    del inner_boundary
    # print("inner_boundary: {}".format(inner_boundary))
    semifinal_inner_boundary = (
        clamp_to_bounding_box(
            expanded_inner_boundary,
            clamped_outer_boundary,
        )
        if clamp_inner_boundary_to_outer_boundary
        else expanded_inner_boundary
    )
    # print("semifinal_inner_boundary: {}".format(semifinal_inner_boundary))
    final_inner_boundary = (
        (semifinal_inner_boundary[2:] + semifinal_inner_boundary[:2])
        if swap_inner_boundary_corners
        else semifinal_inner_boundary
    )
    # print("final_inner_boundary: {}".format(final_inner_boundary))

    if symmetric_hole_and_outer:
        h, o = [None] * 4, [None] * 4
        for i in range(4):
            l, r = make_symmetric_1d(
                original=original_box[i],
                left=clamped_outer_boundary[i] if i < 2 else final_inner_boundary[i],
                right=final_inner_boundary[i] if i < 2 else clamped_outer_boundary[i],
            )
            h[i], o[i] = (r, l) if i < 2 else (l, r)
        sym_hole_border = tuple(h)
        sym_outer_border = tuple(o)
    else:
        sym_hole_border = final_inner_boundary
        sym_outer_border = clamped_outer_boundary
    del final_inner_boundary
    del clamped_outer_boundary

    obj2 = copy.deepcopy(obj)
    (
        x_min_element.text,
        y_min_element.text,
        x_max_element.text,
        y_max_element.text,
    ) = map(
        str,
        sym_hole_border,
    )
    if sym_hole_border == sym_outer_border and reuse_element_if_one_box:
        return
    name_element.text = "{} {} i".format(name, tag)

    box_element2 = obj2.find("bndbox")
    (
        box_element2.find("xmin").text,
        box_element2.find("ymin").text,
        box_element2.find("xmax").text,
        box_element2.find("ymax").text,
    ) = map(
        str,
        sym_outer_border,
    )
    obj2.find("name").text = "{} {} o".format(name, tag)
    root.append(obj2)


def read_filelist(filelist, max_files):
    with open(filelist, mode="r", encoding="UTF-8") as f:
        read_lines = [
            line.strip() for line in itertools.islice(f, 0, max_files) if line.strip()
        ]
    return read_lines
