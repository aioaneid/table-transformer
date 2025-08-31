import functools
from PIL import Image, ImageDraw
import torch
import json
import matplotlib.pyplot as plt
import pathlib
import argparse
import os
import numpy as np
import sys
import itertools
from PIL import Image
from matplotlib import pyplot as plt
import cross_box_plots
import table_plots
from typing import Literal

sys.path.append("src")
import table_datasets


sys.path.append("detr")
import eval

sys.path.append("relax")
import relax_constraints

_DPI = 300
_DEBUG_IMAGE_FILE_EXTENSION = ".png"
_PADDING = 1

structure_class_map = {
    k: v
    for v, k in enumerate(
        [
            "table",
            "table column",
            "table row",
            "table column header",
            "table projected row header",
            "table spanning cell",
            "no object",
        ]
    )
}
detection_class_map = {"table": 0, "table rotated": 1, "no object": 2}


def main(
    input_structure_pascal_voc_xml,
    data_type: Literal["detection", "structure"],
    input_image_dir,
    input_words_data_dir,
    background_image_extension,
    output_image_dir,
):
    os.makedirs(output_image_dir, exist_ok=True)
    xml_size, pairs = table_datasets.read_size_and_label_box_pairs_from_pascal_voc(
        input_structure_pascal_voc_xml
    )
    print(pairs)

    save_filepath = output_image_dir.joinpath(
        output_image_dir.joinpath(
            input_structure_pascal_voc_xml.name.replace(
                ".xml", _DEBUG_IMAGE_FILE_EXTENSION
            )
        )
    )

    labels, bboxes = zip(*pairs)
    int_labels = [
        (
            eval.detection_class_map
            if data_type == "detection"
            else eval.structure_class_map
        )[label]
        for label in labels
    ]

    background_img_filename = input_image_dir.joinpath(
        input_structure_pascal_voc_xml.name.replace(".xml", background_image_extension)
    )

    with Image.open(background_img_filename) as background_img:
        print(
            "Background Image:",
            background_img_filename,
            "ImageSize:",
            background_img.size,
        )

        fig, ax = plt.subplots(1)
        ax.imshow(background_img, interpolation="none")

        for rect in table_plots.get_rectangles(
            table_plots.scale_bbox(
                bboxes,
                background_img.size[0] / xml_size[0],
                background_img.size[1] / xml_size[1],
            ),
            int_labels,
            [1] * len(int_labels),
            data_type,
            _PADDING,
        ):
            ax.add_patch(rect)

    fig.set_size_inches((15, 15))
    plt.axis("off")
    plt.savefig(save_filepath, bbox_inches="tight", dpi=_DPI)
    plt.close()
    print("Saved:", save_filepath)

    if input_words_data_dir:
        words_filepath = args.input_words_data_dir.joinpath(
            "{}_words.json".format(input_structure_pascal_voc_xml.stem)
        )
        with open(words_filepath, "r") as jf:
            tokens = json.load(jf)
        _, original_cells, _ = relax_constraints.eval_cells(
            np.array(bboxes), labels, tokens
        )
        fig, ax = plt.subplots(1)
        ax.imshow(background_img, interpolation="none")

        for cell in original_cells:
            for rect in table_plots.get_cell_rectangles(
                cell,
                background_img.size[0] / xml_size[0],
                background_img.size[1] / xml_size[1],
                _PADDING,
            ):
                ax.add_patch(rect)

        cells_out_filepath = output_image_dir.joinpath(
            "{}_cells{}".format(
                input_structure_pascal_voc_xml.stem, _DEBUG_IMAGE_FILE_EXTENSION
            )
        )

        fig.set_size_inches((15, 15))
        plt.axis("off")
        plt.savefig(cells_out_filepath, bbox_inches="tight", dpi=_DPI)
        plt.close()
        print("Saved:", cells_out_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process tables")
    parser.add_argument(
        "--input_structure_pascal_voc_xml",
        nargs="+",
        type=pathlib.Path,
        help="Comma-separated list of annotation files.",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="structure",
    )
    parser.add_argument(
        "--input_image_dir",
        type=pathlib.Path,
        help="Where to find the images.",
    )
    parser.add_argument(
        "--input_words_data_dir",
        help="Root directory for source data to process",
        type=pathlib.PurePath,
    )
    parser.add_argument(
        "--background_image_extension",
        type=str,
        default=".png",
    )
    parser.add_argument(
        "--output_image_dir",
        type=pathlib.PurePath,
        help="Where to output whole image annotated structure images.",
    )
    parser.add_argument(
        "--print0",
        action="store_true",
        help="If true, print a null character after processing each file.",
    )
    args = parser.parse_args()

    print("Images: {}".format(len(args.input_structure_pascal_voc_xml)), flush=True)
    for input_structure_pascal_voc_xml in args.input_structure_pascal_voc_xml:
        main(
            input_structure_pascal_voc_xml,
            args.data_type,
            args.input_image_dir,
            args.input_words_data_dir,
            args.background_image_extension,
            args.output_image_dir,
        )
        if args.print0:
            print("\x00", flush=True)
