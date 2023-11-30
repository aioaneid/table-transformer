import functools
from PIL import Image, ImageDraw
import torch
import matplotlib.pyplot as plt
import pathlib
import argparse
import os
import cv2
import numpy as np
import math
import distinctipy
import ast
import annotations
import enum
import sys
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch

sys.path.append("src")
import table_datasets

sys.path.append("detr/util")
import cross_box_plots


def main(input_structure_pascal_voc_xml, input_image_dir, output_image_dir):
    os.makedirs(output_image_dir, exist_ok=True)
    pairs = table_datasets.read_label_box_pairs_from_pascal_voc(
        input_structure_pascal_voc_xml
    )
    pairs = table_datasets.convert_label_box_pairs_to_bounds(pairs)
    print(pairs)
    image_path = input_image_dir.joinpath(
        input_structure_pascal_voc_xml.name.replace(".xml", ".jpg")
    )
    img = Image.open(image_path)
    print(img.size)
    w, h = img.size

    for i, (label, bounds) in enumerate(pairs):
        ax = plt.gca()
        ax.imshow(img, interpolation="lanczos")
        plt.axis("off")
        assert len(bounds) == 8
        hole_border = torch.tensor(bounds[:4])
        outer_border = torch.tensor(bounds[4:])
        border_boxes = cross_box_plots.compute_border_boxes(
            outer_border, hole_border, (0, w), (0, h)
        )
        rectangles = cross_box_plots.border_box_rectangles(border_boxes, False)
        print(rectangles)
        for rectangle in rectangles:
            rectangle.set(zorder=1)
            ax.add_patch(rectangle)
        save_filepath = output_image_dir.joinpath(
            input_structure_pascal_voc_xml.name.replace(
                ".xml",
                "_{}_relax_{}_{}.svg".format(
                    input_structure_pascal_voc_xml.parent.name,
                    i,
                    label.replace(" ", "-"),
                ),
            )
        )
        print("Writing: {}".format(save_filepath))
        plt.savefig(save_filepath, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process tables")
    parser.add_argument(
        "--input_structure_pascal_voc_xml",
        nargs="+",
        type=pathlib.Path,
        help="Comma-separated list of annotation files.",
    )
    parser.add_argument(
        "--input_image_dir",
        type=pathlib.Path,
        help="Where to find the images.",
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
            input_structure_pascal_voc_xml, args.input_image_dir, args.output_image_dir
        )
        if args.print0:
            print("\x00", flush=True)
