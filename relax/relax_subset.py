import pathlib
import argparse
import xml.etree.ElementTree as ET
import pathlib
import random
import relax
import numpy as np
import itertools
import annotations


def replace_suffix(s, suffix, replacement):
    if not s.endswith(suffix):
        return s
    return s[: -len(suffix)] + replacement


def output_dataset_name(dataset_name, max_files, remove_non_selected, k, complement: bool):
    return "{}_head_{}_{}_{}{}".format(
        dataset_name, max_files, "only" if remove_non_selected else "kept", k, "_complement" if complement else ""
    )


def main():
    parser = argparse.ArgumentParser(description="Process tables")
    parser.add_argument(
        "--input_pascal_voc_filelist",
        type=pathlib.PurePath,
        help="Text file containing the list of files in format input_pascal_voc, e.g. train_filelist.txt.",
    )
    parser.add_argument("--max_files", type=int, default=2147483647)
    parser.add_argument(
        "--output_dir_root",
        type=pathlib.PurePath,
        help="Where to output the reduced training sets.",
    )
    parser.add_argument(
        "--seed", type=int, help="Where to output the reduced training sets."
    )
    parser.add_argument("--shuf_filelist", type=pathlib.Path, help="Optional.")
    parser.add_argument(
        "--max_shuf_files", type=int, default=2147483647, help="Optional."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=2147483647)
    parser.add_argument("--step", type=int, default=1, help="Should be positive.")
    parser.add_argument("--ks", type=int, nargs="*")
    parser.add_argument("--kept_objects", type=int, nargs="*")
    parser.add_argument(
        "--remove_non_selected", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--anonymize_inner_border", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--complement", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()

    read_lines = relax.read_filelist(args.input_pascal_voc_filelist, args.max_files)
    # dict maintains iteration order, set does not
    lines = {line: None for line in read_lines}.keys()
    assert len(lines) == len(read_lines)
    del read_lines
    print("Lines read: {}".format(len(lines)))

    dir_path = args.input_pascal_voc_filelist.parent

    if args.shuf_filelist:
        with open(args.shuf_filelist, mode="r", encoding="UTF-8") as f:
            read_shuf_line_lengths = [
                (
                    line.strip(),
                    len(
                        ET.parse(dir_path.joinpath(line.strip()))
                        .getroot()
                        .findall("object")
                    ),
                )
                for line in itertools.islice(f, 0, args.max_shuf_files)
                if line.strip() in lines
            ]
        shuf_line_lengths = {
            line_length: None for line_length in read_shuf_line_lengths
        }.keys()
        assert len(shuf_line_lengths) == len(read_shuf_line_lengths)
        assert len(shuf_line_lengths) == len(lines)
        del read_shuf_line_lengths
        shuf_line_lengths = list(shuf_line_lengths)

    ks = args.ks or range(args.start, args.stop, args.step)
    dataset_name = replace_suffix(args.input_pascal_voc_filelist.stem, "_filelist", "")
    if args.kept_objects:
        kept_objects = args.kept_objects
    else:
        kept_objects = []
        rnd = random.Random(args.seed)
        for k in ks:
            object_count = 0
            print("k: {}".format(k))
            output_pascal_voc_dir_name = output_dataset_name(
                dataset_name, args.max_files, args.remove_non_selected, k, args.complement
            )
            if args.output_dir_root:
                output_pascal_voc_dir = pathlib.Path(
                    args.output_dir_root.joinpath(output_pascal_voc_dir_name)
                )
                output_pascal_voc_dir.mkdir(exist_ok=True, parents=True)
            cut_any = False
            for line in lines:
                input_pascal_voc_xml_file = dir_path.joinpath(line)
                if args.output_dir_root:
                    output_pascal_voc_xml_file = output_pascal_voc_dir.joinpath(
                        input_pascal_voc_xml_file.name
                    )
                root = ET.parse(input_pascal_voc_xml_file).getroot()
                size_element = root.find("size")
                width = float(size_element.find("width").text)
                height = float(size_element.find("height").text)
                objects = root.findall("object")
                to_keep = min(len(objects), k)
                if args.complement:
                    to_keep = len(objects) - to_keep
                object_count += to_keep
                # print(objects, len(objects), k, len(objects) <= k)
                if len(objects) <= to_keep:
                    if args.output_dir_root:
                        if output_pascal_voc_xml_file.exists():
                            if (
                                output_pascal_voc_xml_file.readlink()
                                == input_pascal_voc_xml_file
                            ):
                                print(
                                    "Symlink already correct: {}".format(
                                        output_pascal_voc_xml_file.name
                                    )
                                )
                            else:
                                print(
                                    "Removing: {}".format(
                                        output_pascal_voc_xml_file.name
                                    )
                                )
                                output_pascal_voc_xml_file.unlink(missing_ok=True)
                        else:
                            print(
                                "Creating symlink: {}".format(
                                    output_pascal_voc_xml_file.name
                                )
                            )
                            output_pascal_voc_xml_file.symlink_to(
                                input_pascal_voc_xml_file
                            )
                else:
                    cut_any = True
                    indices = set(rnd.sample(range(len(objects)), to_keep))
                    print(
                        "{}: kept {} out of {}".format(
                            input_pascal_voc_xml_file.name, indices, len(objects)
                        )
                    )
                    if args.output_dir_root:
                        for index, obj in enumerate(objects):
                            if index not in indices:
                                if args.remove_non_selected:
                                    root.remove(obj)
                                    continue
                                relax.modify_object_in_place(
                                    root,
                                    index,
                                    obj,
                                    outer_multiplier_and_constant_list=((np.inf, 0),)
                                    * 4,
                                    outer_boundary_clamp_box=(0, 0, width, height),
                                    inner_multiplier_and_constant_list=(
                                        (
                                            (
                                                np.inf
                                                if args.anonymize_inner_border
                                                else 1
                                            ),
                                            0,
                                        ),
                                    )
                                    * 4,
                                    inner_boundary_enclose_box=(
                                        np.inf,
                                        np.inf,
                                        -np.inf,
                                        -np.inf,
                                    ),
                                    clamp_inner_boundary_to_outer_boundary=args.anonymize_inner_border,
                                    swap_inner_boundary_corners=True,
                                    reuse_element_if_one_box=False,
                                    symmetric_hole_and_outer=False,
                                )
                        print("Writing: {}".format(output_pascal_voc_xml_file))
                        annotations.save_xml_pascal_voc(
                            root, output_pascal_voc_xml_file
                        )
            if args.output_dir_root:
                with open(
                    args.output_dir_root.joinpath(
                        "{}_filelist.txt".format(output_pascal_voc_dir_name)
                    ),
                    mode="w",
                    encoding="UTF-8",
                ) as f:
                    for line in lines:
                        f.write(
                            "{}/{}\n".format(
                                output_pascal_voc_dir_name, pathlib.PurePath(line).name
                            )
                        )
            kept_objects.append(object_count)
            if not cut_any:
                break
    print(
        "kept_objects ({}): {}".format(
            len(kept_objects), " ".join(list(map(str, kept_objects)))
        )
    )
    if args.shuf_filelist:
        i = 0
        cum_objects = 0
        for k, ko in zip(
            ks,
            kept_objects,
            strict=False,
        ):
            while i < len(shuf_line_lengths) and cum_objects < ko:
                cum_objects += shuf_line_lengths[i][1]
                i += 1
            while i < len(shuf_line_lengths) and not shuf_line_lengths[i][1]:
                i += 1

            output_filelist_name = "{}_files_{}_desired_objects_{}_cumulative_objects_{}_filelist.txt".format(
                output_dataset_name(
                    dataset_name, args.max_files, args.remove_non_selected, k, args.complement
                ),
                i,
                ko,
                cum_objects,
            )
            print(
                "files: {} with objects: {} >= {} objects into: {}".format(
                    i, cum_objects, ko, output_filelist_name
                )
            )
            if args.output_dir_root:
                output_filelist_path = pathlib.Path(
                    args.output_dir_root.joinpath(output_filelist_name)
                )
                print("Writing: {}".format(output_filelist_path))
                with open(
                    output_filelist_path,
                    mode="w",
                    encoding="UTF-8",
                ) as f:
                    for line, _ in itertools.islice(shuf_line_lengths, 0, i):
                        f.write("{}\n".format(line))


if __name__ == "__main__":
    main()
