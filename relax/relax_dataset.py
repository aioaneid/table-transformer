import pathlib
import argparse
import os
import xml.etree.ElementTree as ET
import relax
import numpy as np
from enum_actions import enum_action
import annotations


def main():
    parser = argparse.ArgumentParser(description="Process tables")
    parser.add_argument(
        "--input_pascal_voc_xml_files",
        nargs="*",
        help="Comma-separated list of input Pascal POC XML files.",
    )
    parser.add_argument(
        "--output_pascal_voc_xml_dir",
        help="Where to output pascal voc XML files.",
    )
    parser.add_argument(
        "--inner_multiplier",
        type=float,
        help="Usually <= 1; inner boundary will have length width * inner_multiplier and be centered. If negative that will have the effect of marking the inner boundary as missing, unless the boundary was empty to start with, in which case the point it represents is reflected on both axes. Can also be '[+/-]inf'.",
        default=[1] * 4,
        nargs=4,
    )
    parser.add_argument(
        "--inner_constant",
        type=float,
        help="Usually <= 0; inner boundary will have length (width + inner_constant) * inner_multiplier and be centered.",
        default=[0] * 4,
        nargs=4,
    )
    parser.add_argument(
        "--outer_multiplier",
        type=float,
        help="Usually >= 1; outer boundary will have length width * outer_multiplier and be centered.  If negative that will have the effect of marking the inner boundary as missing, unless the boundary was empty to start with, in which case the point it represents is reflected on both axes. Can also be '[+/-]inf'.",
        default=[1] * 4,
        nargs=4,
    )
    parser.add_argument(
        "--outer_constant",
        type=float,
        help="Usually >= 0; outer boundary will have length (width + outer_constant) * outer_multiplier and be centered.",
        default=[0] * 4,
        nargs=4,
    )
    parser.add_argument(
        "--print0",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, print a null character after processing each file.",
    )
    parser.add_argument(
        "--clamp_outer_boundary",
        action=enum_action(relax.OuterClamp),
        default=relax.OuterClamp.NONE,
    )
    parser.add_argument(
        "--enclose_inner_boundary",
        action=enum_action(relax.InnerEnclose),
        default=relax.InnerEnclose.NONE,
    )
    parser.add_argument(
        "--clamp_inner_boundary_to_outer_boundary",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to clamp inner boundary to (streched and clamped) outer boundary.",
    )
    parser.add_argument(
        "--swap_inner_boundary_corners",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to swap inner boundary [:2] with [2:]. Runs after clamp_inner_boundary_to_outer_boundary.",
    )
    parser.add_argument(
        "--reuse_element_if_one_box",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to reuse the element if outer is identical to inner.",
    )
    parser.add_argument(
        "--symmetric_hole_and_outer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to make sure that the original borders are exactly in the middle of the computed hole and outer borders.",
    )
    args = parser.parse_args()

    print("Images: {}".format(len(args.input_pascal_voc_xml_files)), flush=True)
    for input_pascal_voc_xml_file in args.input_pascal_voc_xml_files:
        os.makedirs(args.output_pascal_voc_xml_dir, exist_ok=True)
        pure_posix_path = pathlib.PurePosixPath(input_pascal_voc_xml_file)
        tree = ET.parse(input_pascal_voc_xml_file)
        modify_in_place(
            list(zip(args.outer_multiplier, args.outer_constant)),
            list(zip(args.inner_multiplier, args.inner_constant)),
            args.clamp_outer_boundary,
            args.enclose_inner_boundary,
            args.clamp_inner_boundary_to_outer_boundary,
            args.swap_inner_boundary_corners,
            args.reuse_element_if_one_box,
            symmetric_hole_and_outer=args.symmetric_hole_and_outer,
            tree=tree,
        )
        output_file_path = os.path.join(
            args.output_pascal_voc_xml_dir, pure_posix_path.name
        )
        print(output_file_path)
        tree.write(output_file_path)
        if args.print0:
            print("\x00", flush=True)


def enclosing_box(a, b):
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def outer_boundary_clamp_box(
    clamp_outer_boundary, image_box, original_table_box, original_box
):
    match clamp_outer_boundary:
        case relax.OuterClamp.NONE:
            return (-np.inf, -np.inf, np.inf, np.inf)
        case relax.OuterClamp.IMAGE:
            return image_box
        case relax.OuterClamp.TABLE:
            return original_table_box
        case relax.OuterClamp.ORIGINAL_OR_TABLE:
            return enclosing_box(original_box, original_table_box)


def inner_boundary_enclose_box(
    enclose_inner_boundary, original_table_box, original_box
):
    match enclose_inner_boundary:
        case relax.InnerEnclose.NONE:
            return (np.inf, np.inf, -np.inf, -np.inf)
        case relax.InnerEnclose.ORIGINAL_NEAR_TABLE_EDGES:
            return tuple(
                (
                    original_box[i]
                    if abs(original_table_box[i] - original_box[i]) < 1
                    else (np.inf if i < 2 else -np.inf)
                )
                for i in range(4)
            )


def find_table_object(objects):
    for obj in objects:
        name_element = obj.find("name")
        if name_element.text in ["table", "table rotated"]:
            return obj
    assert False, [obj.find("name").text for obj in objects]


def modify_in_place(
    outer_multiplier_and_constant_list,
    inner_multiplier_and_constant_list,
    clamp_outer_boundary,
    enclose_inner_boundary,
    clamp_inner_boundary_to_outer_boundary,
    swap_inner_boundary_corners,
    reuse_element_if_one_box,
    *,
    symmetric_hole_and_outer: bool,
    tree,
):
    root = tree.getroot()
    size_element = root.find("size")
    image_box = (
        0,
        0,
        float(size_element.find("width").text),
        float(size_element.find("height").text),
    )
    objects = root.findall("object")
    original_table_box = relax.read_bbox_of_object(find_table_object(objects))

    for index, obj in enumerate(objects):
        original_box = annotations.get_bounding_box_of_object(obj)
        relax.modify_object_in_place(
            root,
            index,
            obj,
            outer_multiplier_and_constant_list,
            outer_boundary_clamp_box(
                clamp_outer_boundary, image_box, original_table_box, original_box
            ),
            inner_multiplier_and_constant_list,
            inner_boundary_enclose_box(
                enclose_inner_boundary, original_table_box, original_box
            ),
            clamp_inner_boundary_to_outer_boundary,
            swap_inner_boundary_corners,
            reuse_element_if_one_box,
            symmetric_hole_and_outer=symmetric_hole_and_outer,
        )


if __name__ == "__main__":
    main()
