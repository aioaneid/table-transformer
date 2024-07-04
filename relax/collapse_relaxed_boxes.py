import pathlib
import argparse
import xml.etree.ElementTree as ET
import pathlib
import annotations
import os
import sys
import collections

sys.path.append("src")
import table_datasets


def replace_suffix(s, suffix, replacement):
    if not s.endswith(suffix):
        return s
    return s[: -len(suffix)] + replacement


def output_dataset_name(
    dataset_name, max_files, remove_non_selected, k, complement: bool
):
    return "{}_head_{}_{}_{}{}".format(
        dataset_name,
        max_files,
        "only" if remove_non_selected else "kept",
        k,
        "_complement" if complement else "",
    )


def read_object_pair_dict(root):
    d = collections.defaultdict(lambda: collections.defaultdict(lambda: [None, None]))
    for obj in root.iter("object"):
        label = obj.find("name").text
        m = table_datasets.HOLE_OR_OUTER_LABEL.fullmatch(label)
        assert m
        assert m.group(3) in ["i", "o"]
        outside = m.group(3) == "o"
        io = d[m.group(1)][m.group(2)]
        assert not io[outside]
        io[outside] = obj
    for label, id2io in d.items():
        for _, io in id2io.items():
            assert len(io) == 2
            assert io[0]
            assert io[1]
    return d


def main():
    parser = argparse.ArgumentParser(description="Process tables")
    parser.add_argument(
        "--input_pascal_voc_xml_files",
        nargs="*",
        help="Space-separated list of input Pascal POC XML files.",
        type=pathlib.PurePath,
    )
    parser.add_argument(
        "--output_pascal_voc_xml_dir",
        help="Where to output pascal voc XML files.",
        type=pathlib.PurePath,
    )
    parser.add_argument(
        "--skip_if_output_exists",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--weight_hole", type=float, default=1)
    parser.add_argument("--weight_outer", type=float, default=1)
    parser.add_argument(
        "--print0",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, print a null character after processing each file.",
    )
    args = parser.parse_args()

    print({**vars(args), **{"input_pascal_voc_xml_files": None}})

    if args.input_pascal_voc_xml_files:
        os.makedirs(args.output_pascal_voc_xml_dir, exist_ok=True)
    for input_pascal_voc_xml_file in args.input_pascal_voc_xml_files:
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
        tree = ET.parse(str(input_pascal_voc_xml_file))
        root = tree.getroot()
        d = read_object_pair_dict(root)
        for label, id2io in d.items():
            for _, io in id2io.items():
                hole_elements = annotations.get_bounding_box_elements(io[0])
                hole_box = list(
                    annotations.get_bounding_box_of_elements(*hole_elements)
                )
                outer_elements = annotations.get_bounding_box_elements(io[1])
                outer_box = list(
                    annotations.get_bounding_box_of_elements(*outer_elements)
                )
                weighted_average_box = [
                    (args.weight_hole * h + args.weight_outer * o)
                    / (args.weight_hole + args.weight_outer)
                    for (h, o) in zip(hole_box, outer_box)
                ]
                io[0].find("name").text = label
                for i in range(4):
                    hole_elements[i].text = str(weighted_average_box[i])
                root.remove(io[1])

        print(output_file_path)
        tree.write(output_file_path)
        if args.print0:
            print("\x00", flush=True)


if __name__ == "__main__":
    main()
