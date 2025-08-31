import pathlib
import argparse
import os
import fitz
import math
import json
import logging
from PIL import Image
import statistics
import annotations

logger = logging.getLogger(__name__)


class ImageArgs(object):
    def __init__(
        self,
        image_dir,
        image_extension,
        output_image_words_dir,
        output_image_text_pascal_voc_xml_dir,
    ):
        self.image_dir = image_dir
        self.image_extension = image_extension
        self.output_image_words_dir = output_image_words_dir
        self.output_image_text_pascal_voc_xml_dir = output_image_text_pascal_voc_xml_dir


def scale_box(box, scaling_factor):
    return tuple(x * scaling_factor for x in box)


def process_image(words_stem, d, image_args):
    image_filename = "{}{}".format(words_stem, image_args.image_extension)
    image_path = image_args.image_dir.joinpath(image_filename)
    if not annotations.image_exists_and_is_valid(image_path):
        logger.warning("Missing image: %s", image_path)
        return
    with Image.open(image_path) as im:
        im_w, im_h = im.size
    pdf_w = d["pdf_page_rect"][2] - d["pdf_page_rect"][0]
    pdf_h = d["pdf_page_rect"][3] - d["pdf_page_rect"][1]
    ratio_deviation = im_w / im_h - pdf_w / pdf_h
    logger.info("ratio_deviation: %f", ratio_deviation)
    if abs(ratio_deviation) > 1e-2:
        raise Exception("ratio_deviation: {}".format(ratio_deviation))
    scaling_factor = statistics.mean([im_w / pdf_w, im_h / pdf_h])
    image_d = {
        **d,
        "image_rect": [0, 0, im_w, im_h],
        "words": [
            {**word, "bbox": scale_box(word["bbox"], scaling_factor)}
            for word in d["words"]
        ],
    }
    del d
    os.makedirs(image_args.output_image_words_dir, exist_ok=True)
    json_path = image_args.output_image_words_dir.joinpath(
        "{}_words.json".format(image_path.stem)
    )
    logging.info(json_path)
    with open(json_path, "wt") as out_file:
        json.dump(image_d, out_file, sort_keys=True, indent=4)
    if image_args.output_image_text_pascal_voc_xml_dir:
        os.makedirs(image_args.output_image_text_pascal_voc_xml_dir, exist_ok=True)
        annotation = annotations.create_pascal_voc_page_element(
            image_path,
            image_d["image_rect"][0],
            image_d["image_rect"][1],
            "bluecare-image-text",
        )
        for word in image_d["words"]:
            element = annotations.create_pascal_voc_object_element("text", word["bbox"])
            annotation.append(element)
        dest_annot_path = os.path.join(
            image_args.output_image_text_pascal_voc_xml_dir, f"{image_path.stem}.xml"
        )
        annotations.save_xml_pascal_voc(annotation, dest_annot_path)


def transform_box(box, transformation_matrix):
    return sum(
        (
            (p.x, p.y)
            for p in (
                p.transform(transformation_matrix)
                for p in (fitz.Point(box[:2]), fitz.Point(box[2:]))
            )
        ),
        (),
    )


def main(pdf_file, output_words_dir, image_args):
    with fitz.open(pdf_file) as doc:
        for page_index, page in enumerate(doc):
            # https://pymupdf.readthedocs.io/en/latest/textpage.html#TextPage.extractDICT
            text_dict = page.get_text("dict", sort=True)
            spans = [
                {
                    "block_num": block["number"],
                    "flags": span["flags"],
                    "bbox": transform_box(span["bbox"], page.rotation_matrix),
                    "text": span["text"],
                }
                for block in text_dict["blocks"]
                for line in block.get("lines") or []
                for span in line["spans"]
                if block["type"] == 0
            ]

            d = {
                "pdf_page_rect": list(page.rect),  # Unused by model
                # 'image_rect': page.rect,
                "words": [
                    {
                        **span,
                        "span_num": i + 1,
                    }
                    for i, span in enumerate(spans)
                ],
            }
            words_stem = (
                "{}-{:0" + str(math.ceil(math.log(len(doc) + 1, 10))) + "d}"
            ).format(pdf_file.name, page_index + 1)
            json_path = output_words_dir.joinpath("{}_words.json".format(words_stem))
            logging.info(json_path)
            with open(json_path, "wt") as out_file:
                json.dump(d, out_file, sort_keys=True, indent=4)
            if image_args:
                process_image(words_stem, d, image_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process tables")
    parser.add_argument(
        "--pdf_files",
        nargs="+",
        type=pathlib.Path,
        help="Comma-separated list of PDF files with text.",
    )
    parser.add_argument(
        "--output_words_dir",
        type=pathlib.Path,
        help="Where to output the text boxes to json files.",
    )
    parser.add_argument(
        "--image_dir",
        type=pathlib.Path,
        help="Where to find the corresponding images (optional).",
    )
    parser.add_argument(
        "--image_extension",
        type=str,
        help=".ppm, .jpg (optional).",
    )
    parser.add_argument(
        "--output_image_words_dir",
        type=pathlib.Path,
        help="Where to output the text boxes to json files.",
    )
    parser.add_argument(
        "--output_image_text_pascal_voc_xml_dir",
        type=pathlib.Path,
        help="Where to write the pascal voc text boxes. Only if image_dir is also specified.",
    )
    parser.add_argument(
        "--print0",
        action="store_true",
        help="If true, print a null character after processing each file.",
    )
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    logging.info("PDF files: %d", len(args.pdf_files))
    os.makedirs(args.output_words_dir, exist_ok=True)
    for pdf_file in args.pdf_files:
        main(
            pdf_file,
            args.output_words_dir,
            None
            if not args.image_dir
            else ImageArgs(
                args.image_dir,
                args.image_extension,
                args.output_image_words_dir,
                args.output_image_text_pascal_voc_xml_dir,
            ),
        )
        if args.print0:
            print("\x00", flush=True)
    # TODO: Add image dir and scale coordinates so they match the image sizes.
