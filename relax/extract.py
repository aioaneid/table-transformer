import functools
from PIL import Image, ImageDraw
from transformers import DetrImageProcessor
from transformers import TableTransformerForObjectDetection
import torch
import matplotlib.pyplot as plt
import pathlib
import argparse
import os
from xml.dom import minidom
import xml.etree.ElementTree as ET
from fitz import Rect
import cv2
import numpy as np
import math
import distinctipy
import ast
import annotations
import enum


# def add_margin(pil_img, top, right, bottom, left, color):
#     width, height = pil_img.size
#     new_width = width + right + left
#     new_height = height + top + bottom
#     result = Image.new(pil_img.mode, (new_width, new_height), color)
#     result.paste(pil_img, (left, top))
#     return result


# def pad_image(image, margin):
#     return add_margin(image, margin, margin, margin, margin, "white")


# def pad_box(box, margin):
#     a, b, c, d = box
#     return a - margin, b - margin, c + margin, d + margin


def plot_results(ax, scores, labels, boxes):
    linestyles = ["-", "--", "-.", ":"]
    hatches = ["/", "o", "\\", "O", "x", ".", "*"]
    rectangles = [
        plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            color=plt.cm.brg(i / boxes.size(dim=0)),
            linewidth=1,
            ls=linestyles[i % len(linestyles)],
            hatch=hatches[i % len(hatches)],
            alpha=0.5,
        )
        for i, (xmin, ymin, xmax, ymax) in enumerate(boxes.tolist())
    ]
    handles = [ax.add_patch(rectangle) for rectangle in rectangles]
    # leg = ax.legend(
    #     handles=handles,
    #     labels=[
    #         f"{i} {label}: {score:.3f}"
    #         for i, (label, score) in enumerate(zip(labels, scores.tolist()))
    #     ],
    # )
    leg = ax.legend(
        handles=handles,
        labels=[f"{score:.2f}" for score in scores.tolist()],
    )
    # leg = ax.legend([", ".join(f"{score:.2f}" for score in scores.tolist())])
    for _txt in leg.texts:
        _txt.set_alpha(0.5)
    compressed_labels = []
    i = 0
    while i != len(labels):
        j = i + 1
        while j != len(labels) and labels[j] == labels[i]:
            j += 1
        compressed_labels.append((labels[i], j))
        i = j
    ax.set_title(
        " - ".join(["0"] + sum(([label, str(j)] for label, j in compressed_labels), []))
    )

    # for i, (score, scaled_score, _, (xmin, ymin, xmax, ymax)) in enumerate(
    #     zip(scores.tolist(), scaled_scores.tolist(), labels.tolist(), boxes.tolist())
    # ):
    #     c =
    #     ax.add_patch(
    #         plt.Rectangle(
    #             (xmin, ymin),
    #             xmax - xmin,
    #             ymax - ymin,
    #             fill=False,
    #             color=plt.cm.brg(scaled_score),
    #             linewidth=2,
    #             ls=linestyles[i % len(linestyles)],
    #             hatch=hatches[i % len(hatches)],
    #         )
    #     )
    #     text = f"{score:0.2f}"
    #     ax.text(
    #         xmin,
    #         (ymin + ymax) / 2,
    #         text,
    #         fontsize=15,
    #         color=c,
    #         horizontalalignment="right",
    #     )


# def analyze_structure(feature_extractor, smodel, table_image, file_prefix):
#     encoding = feature_extractor(table_image, return_tensors="pt")
#     with torch.no_grad():
#         soutputs = smodel(**encoding)

#     target_sizes = [table_image.size[::-1]]
#     sresults = feature_extractor.post_process_object_detection(
#         soutputs, threshold=0.7, target_sizes=target_sizes
#     )[0]
#     for label in smodel.config.id2label.keys():
#         indicator = sresults["labels"] == label
#         splt = plot_results(
#             table_image,
#             sresults["scores"][indicator],
#             sresults["labels"][indicator],
#             sresults["boxes"][indicator],
#         )
#         structure_png = f"{file_prefix}_{label}_{smodel.config.id2label[label]}.png"
#         splt.savefig(structure_png, bbox_inches="tight")
#         print(structure_png)


def line_from_contour(contour):
    POLYGON_APPROXIMATION_PERIMETER_FACTOR = 0.02
    epsilon = (
        cv2.arcLength(curve=contour, closed=True)
        * POLYGON_APPROXIMATION_PERIMETER_FACTOR
    )
    approximations = cv2.approxPolyDP(curve=contour, epsilon=epsilon, closed=True)
    assert len(approximations.shape) == 3
    if approximations.shape[:2] != (2, 1):
        print("{} is not a line".format(approximations))
        return None
    return np.squeeze(approximations, axis=1)


SEGMENT_LENGTH_FACTOR = 0.75


def reduce_length_keeping_center(x, y):
    a = (x + y) / 2
    b = (y - x) / 2
    return a - b * SEGMENT_LENGTH_FACTOR, a + b * SEGMENT_LENGTH_FACTOR


ALLOWED_DISTANCE_FACTOR = 1 / 4

# mb must not be much larger in any dimension, otherwise this will do nothing at all
def trust_first_adjust_second(fb, mb):
    assert fb.size() == (4,)
    assert mb.size() == (4,)

    mrl = reduce_length_keeping_center(mb[0], mb[2])
    if fb[0::2].min().item() <= min(mrl) and max(mrl) <= fb[0::2].max().item():
        for i in range(1, 4, 2):
            for j in range(1, 4, 2):
                if (
                    abs(fb[i] - mb[j])
                    <= min(abs(fb[1] - fb[3]), abs(mb[1] - mb[3]))
                    * ALLOWED_DISTANCE_FACTOR
                ):
                    mb[j] = fb[i]
    mrl = reduce_length_keeping_center(mb[1], mb[3])
    if fb[1::2].min().item() <= min(mrl) and max(mrl) <= fb[1::2].max().item():
        for i in range(0, 3, 2):
            for j in range(0, 3, 2):
                if (
                    abs(fb[i] - mb[j])
                    <= min(abs(fb[0] - fb[2]), abs(mb[0] - mb[2]))
                    * ALLOWED_DISTANCE_FACTOR
                ):
                    mb[j] = fb[i]


# Similar to get_class_map in inference.py but with different ordering and a new entry "table rotated".
# The order is used by output_whole_image_sanitized_structure_pascal_voc_xml_dir.
class WholeImageStructureLabel(enum.Enum):
    TABLE = enum.auto()  # starts at 1
    TABLE_ROTATED = enum.auto()  # very important
    TABLE_COLUMN = enum.auto()
    TABLE_COLUMN_HEADER = enum.auto()
    TABLE_PROJECTED_ROW_HEADER = enum.auto()
    TABLE_ROW = enum.auto()
    TABLE_SPANNING_CELL = enum.auto()

    def external(self):
        return self.name.lower().replace("_", " ")

    def rotate(self):
        match self:
            case WholeImageStructureLabel.TABLE:
                return WholeImageStructureLabel.TABLE_ROTATED
            case WholeImageStructureLabel.TABLE_ROTATED:
                raise ValueError("We should never have to rotate a table twice")
            case _:
                return self

    @classmethod
    def from_external(cls, s):
        return cls[s.upper().replace(" ", "_")]


def rotate_whole_image_structure_label(e, rotate):
    return e.rotate() if rotate else e


def to_whole_image_structure_label_tensor(slabel_tensor, rotated, smodel_id2label):
    return slabel_tensor.apply_(
        lambda slabel: rotate_whole_image_structure_label(
            WholeImageStructureLabel.from_external(smodel_id2label[slabel]), rotated
        ).value
    )


def main(
    feature_extractor,
    detection_model,
    smodel,
    input_page_image_file_path,
    annotated_detection_image_dir,
    output_detection_pascal_voc_xml_dir,
    input_detection_pascal_voc_xml_dir,
    distance_to_tables_for_line_removal,
    line_width,
    min_removed_lines,
    reprocess_removed_lines,
    output_image_without_lines_dir,
    output_image_with_contours_dir,
    horizontal_structuring_kernel_length,
    vertical_structuring_kernel_length,
    line_colors,
    structure_table_padding,
    cropped_table_dir,
    output_table_cropped_pascal_voc_xml_dir,
    output_annotated_structure_image_dir,
    output_structure_pascal_voc_xml_dir,
    output_whole_image_sanitized_structure_pascal_voc_xml_dir,
    use_detection_tables_in_whole_image_structure,
    output_whole_image_structure_pascal_voc_xml_dir,
    input_whole_image_structure_pascal_voc_xml_dir,
    output_whole_image_annotated_structure_image_dir,
):
    raw_image = Image.open(input_page_image_file_path)
    rgb_image = raw_image.convert("RGB")

    pure_posix_path = pathlib.PurePosixPath(input_page_image_file_path)

    detection_id2label = {0: "table", 1: "table rotated"}
    detection_label2id = dict((value, key) for key, value in detection_id2label.items())

    if output_detection_pascal_voc_xml_dir or not input_detection_pascal_voc_xml_dir:
        encoding = feature_extractor(rgb_image, return_tensors="pt")
        assert detection_model.config.id2label == detection_id2label

        with torch.no_grad():
            outputs = detection_model(**encoding)

        # rescale bounding boxes
        width, height = rgb_image.size
        detection_result = feature_extractor.post_process_object_detection(
            outputs, threshold=0.9, target_sizes=[(height, width)]
        )[0]

        detection_labels = detection_result["labels"]
        detection_boxes = detection_result["boxes"]
        detection_scores = detection_result["scores"]
        del detection_result

    if output_detection_pascal_voc_xml_dir:
        os.makedirs(output_detection_pascal_voc_xml_dir, exist_ok=True)
        annotation = annotations.create_pascal_voc_page_element(
            pure_posix_path, width, height, "custom-detection"
        )
        for label, box in zip(detection_labels.tolist(), detection_boxes.unbind()):
            element = annotations.create_pascal_voc_object_element(
                detection_id2label[label], box.numpy()
            )
            annotation.append(element)
        dest_annot_path = os.path.join(
            output_detection_pascal_voc_xml_dir, f"{pure_posix_path.stem}.xml"
        )
        print(dest_annot_path)
        annotations.save_xml_pascal_voc(annotation, dest_annot_path)

    table_id = detection_label2id["table"]
    table_rotated_id = detection_label2id["table rotated"]
    if input_detection_pascal_voc_xml_dir:
        in_annot_path = os.path.join(
            input_detection_pascal_voc_xml_dir, f"{pure_posix_path.stem}.xml"
        )
        input_boxes, input_labels = annotations.read_pascal_voc(in_annot_path)
        table_input_boxes = [
            input_box
            for input_box, input_label in zip(input_boxes, input_labels)
            if input_label == "table" or input_label == "table rotated"
        ]
        input_detection_boxes = torch.as_tensor(
            table_input_boxes, dtype=torch.float32
        ).reshape(len(table_input_boxes), 4)
        input_detection_labels = torch.as_tensor(
            [
                detection_label2id[input_label]
                for input_label in input_labels
                if input_label == "table" or input_label == "table rotated"
            ],
            dtype=torch.int64,
        )
        if input_detection_pascal_voc_xml_dir == output_detection_pascal_voc_xml_dir:
            assert torch.equal(input_detection_labels, detection_labels), (
                input_detection_labels,
                detection_labels,
            )
            assert torch.equal(input_detection_boxes, detection_boxes), (
                input_detection_boxes,
                detection_boxes,
            )
        else:
            detection_labels, detection_boxes, detection_scores = (
                input_detection_labels,
                input_detection_boxes,
                torch.ones_like(input_detection_labels, dtype=torch.float32),
            )
        del input_detection_labels, input_detection_boxes

    detection_table_indicator = torch.logical_xor(
        detection_labels == table_id, detection_labels == table_rotated_id
    )

    if annotated_detection_image_dir:
        os.makedirs(annotated_detection_image_dir, exist_ok=True)
        ax = plt.gca()
        ax.imshow(rgb_image)

        plot_results(
            ax,
            detection_scores[detection_table_indicator],
            [
                detection_id2label[label_id]
                for label_id in detection_labels[detection_table_indicator].tolist()
            ],
            detection_boxes[detection_table_indicator],
        )
        plt.gcf().set_size_inches(20 * rgb_image.size[0] / rgb_image.size[1], 20)
        plt.axis("off")
        detection_png = os.path.join(
            annotated_detection_image_dir, f"{pure_posix_path.stem}_detection.png"
        )
        print(detection_png)
        plt.savefig(detection_png, bbox_inches="tight")
        plt.close()

    if output_image_without_lines_dir:
        os.makedirs(output_image_without_lines_dir, exist_ok=True)
        no_lines_path = os.path.join(
            output_image_without_lines_dir, f"{pure_posix_path.stem}.png"
        )
        if not reprocess_removed_lines and annotations.image_exists_and_is_valid(
            no_lines_path
        ):
            print(
                "Skipping {} because it already exists and is valid.".format(
                    no_lines_path
                )
            )
        else:
            # https://stackoverflow.com/questions/35609719/opencv-houghlinesp-parameters
            # Convert the image to gray-scale
            gray = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2GRAY)
            # apertureSize of 7 finds a spurious line under the logo in Spital_Linth_1.pdf-3
            # apertureSize of 3 fails to detect the second half of the header line in same file.
            edges = cv2.Canny(gray, 50, 200, apertureSize=5)

            theta_tolerance = np.pi / 180
            angle_tolerance = np.pi / 180
            # Spital_Linth_1.pdf-3 needs a maxLineGap >= 11 (if apertureSize=3).
            # SpitexWyland_1.pdf-1 wants minLineLength <= 17
            lines = cv2.HoughLinesP(
                edges,
                rho=0.5,
                theta=theta_tolerance,
                threshold=80,
                minLineLength=17,
                maxLineGap=80,
            )
            lines_to_remove = []
            if lines is not None:
                # Draw lines on the image
                for line in lines:
                    # print(line)
                    pt1 = np.asarray(line[0][:2], dtype=int)
                    pt2 = np.asarray(line[0][2:], dtype=int)
                    diff = pt2 - pt1

                    diff_norm = np.linalg.norm(diff)
                    is_horizontal = (
                        abs(diff[1]) <= math.sin(angle_tolerance) * diff_norm
                    )
                    bboxes = [
                        annotations.pad_torch_box(
                            detection_box, distance_to_tables_for_line_removal
                        )
                        for detection_box in detection_boxes
                    ]
                    x_matches = [
                        min(pt1[0], pt2[0]) <= max(bbox[0], bbox[2])
                        and max(pt1[0], pt2[0]) >= min(bbox[0], bbox[2])
                        for bbox in bboxes
                    ]
                    y_matches = [
                        min(pt1[1], pt2[1]) <= max(bbox[1], bbox[3])
                        and max(pt1[1], pt2[1]) >= min(bbox[1], bbox[3])
                        for bbox in bboxes
                    ]
                    is_vertical = (
                        abs(diff[0] <= math.cos(np.pi / 2 - angle_tolerance))
                        * diff_norm
                    )
                    # Do not remove upper line from T.
                    if (
                        (is_horizontal and diff_norm >= 500)
                        or (is_vertical and diff_norm >= 100)
                    ) and any((x_m and y_m for x_m, y_m in zip(x_matches, y_matches))):
                        lines_to_remove.append((pt1, pt2))
                        # cv2.line(no_lines_img, pt1, pt2, (255, 0, 0), 3)
            if len(lines_to_remove) >= min_removed_lines:
                ri = rgb_image.copy()
                draw = ImageDraw.Draw(ri)
                # exclude_colors=None automatically excludes white and black
                colors = line_colors or distinctipy.get_colors(len(lines_to_remove))
                for i, (pt1, pt2) in enumerate(lines_to_remove):
                    print("Line: {} {}".format(pt1, pt2))
                    draw.line(
                        xy=[tuple(pt1), tuple(pt2)],
                        fill=tuple(
                            int(round(c * 255)) for c in colors[i % len(colors)]
                        ),
                        width=line_width,
                    )
                print(no_lines_path)
                ri.save(no_lines_path)
            else:
                print(
                    "Skipping: {} due to not enough lines to remove: {}".format(
                        no_lines_path, lines_to_remove
                    )
                )
    if output_image_with_contours_dir:
        os.makedirs(output_image_with_contours_dir, exist_ok=True)
        no_lines_path = os.path.join(
            output_image_with_contours_dir, f"{pure_posix_path.stem}.png"
        )
        if not reprocess_removed_lines and annotations.image_exists_and_is_valid(
            no_lines_path
        ):
            print(
                "Skipping {} because it already exists and is valid.".format(
                    no_lines_path
                )
            )
        else:
            # See DipTableLineDetector.

            # https://learnopencv.com/contour-detection-using-opencv-python-c/#Steps-for-Finding-and-Drawing-Contours-in-OpenCV
            gray = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2GRAY)

            inverted = cv2.bitwise_not(gray)
            binary = cv2.adaptiveThreshold(
                inverted, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
            )
            # Typical binary.shape for pubmed data is binary.shape: (1000, 759)
            print("binary.shape: {}".format(binary.shape))

            processed_images = []
            if True:
                h_binary = binary.copy()
                H_STRUCTURE_ELEMENT_DIMENSION_SCALE = 30
                h_kernel = cv2.getStructuringElement(
                    shape=cv2.MORPH_RECT,
                    ksize=(
                        horizontal_structuring_kernel_length
                        or h_binary.shape[0] // H_STRUCTURE_ELEMENT_DIMENSION_SCALE,
                        1,
                    ),
                )
                cv2.erode(h_binary, h_kernel, h_binary)
                cv2.dilate(h_binary, h_kernel, h_binary)
                processed_images.append(h_binary)
                del h_binary, h_kernel

            if True:
                v_binary = binary.copy()
                V_STRUCTURE_ELEMENT_DIMENSION_SCALE = 60
                v_kernel = cv2.getStructuringElement(
                    shape=cv2.MORPH_RECT,
                    ksize=(
                        1,
                        vertical_structuring_kernel_length
                        or v_binary.shape[1] // V_STRUCTURE_ELEMENT_DIMENSION_SCALE,
                    ),
                )
                cv2.erode(v_binary, v_kernel, v_binary)
                cv2.dilate(v_binary, v_kernel, v_binary)
                processed_images.append(v_binary)
                del v_binary, v_kernel

            contours = sum(
                (
                    cv2.findContours(
                        image=processed_image,
                        mode=cv2.RETR_EXTERNAL,
                        method=cv2.CHAIN_APPROX_SIMPLE,
                    )[0]
                    for processed_image in processed_images
                ),
                tuple(),
            )
            optional_lines = [line_from_contour(contour) for contour in contours]
            lines_to_remove = [line for line in optional_lines if line is not None]
            print("lines_to_remove: {}".format(lines_to_remove))

            if len(lines_to_remove) >= min_removed_lines:
                ri = rgb_image.copy()
                draw = ImageDraw.Draw(ri)
                # exclude_colors=None automatically excludes white and black
                colors = line_colors or distinctipy.get_colors(len(lines_to_remove))
                print(colors)
                for i, (pt1, pt2) in enumerate(lines_to_remove):
                    print("Line: {} {}".format(pt1, pt2))
                    draw.line(
                        xy=[tuple(pt1), tuple(pt2)],
                        fill=tuple(
                            int(round(c * 255)) for c in colors[i % len(colors)]
                        ),
                        width=line_width,
                    )
                print(no_lines_path)
                ri.save(no_lines_path)
            else:
                print(
                    "Skipping: {} due to not enough lines to remove: {}".format(
                        no_lines_path, lines_to_remove
                    )
                )

    rounded_detection_boxes = detection_boxes[detection_table_indicator].round()
    table_crop_boxes = [
        annotations.pad_torch_box(box, structure_table_padding)
        for box in rounded_detection_boxes
    ]
    raw_cropped_table_images = [
        rgb_image.crop(crop_box.tolist()) for crop_box in table_crop_boxes
    ]
    rotations = (
        detection_labels[detection_table_indicator] == table_rotated_id
    ).tolist()
    cropped_table_images = [
        row_cropped_table_image.transpose(Image.Transpose.ROTATE_270)
        if rotated
        else row_cropped_table_image
        for (row_cropped_table_image, rotated) in zip(
            raw_cropped_table_images,
            rotations,
            strict=True,
        )
    ]
    if cropped_table_dir:
        os.makedirs(cropped_table_dir, exist_ok=True)
        for i, cropped_table_image in enumerate(cropped_table_images):
            cropped_table_output_path = os.path.join(
                cropped_table_dir, f"{pure_posix_path.stem}_{i}.png"
            )
            print(cropped_table_output_path)
            cropped_table_image.save(cropped_table_output_path)

    if output_table_cropped_pascal_voc_xml_dir:
        os.makedirs(output_table_cropped_pascal_voc_xml_dir, exist_ok=True)
        for i, (table_crop_box, detection_label_id) in enumerate(
            zip(
                table_crop_boxes,
                detection_labels[detection_table_indicator].tolist(),
            )
        ):
            assert detection_label_id in set((table_id, table_rotated_id))
            rotated = detection_label_id == table_rotated_id
            # detection_labels, detection_boxes
            table_crop_width, table_crop_height = tuple(
                (table_crop_box[2:] - table_crop_box[:2]).numpy()
            )
            annotation = annotations.create_pascal_voc_page_element(
                pathlib.PurePosixPath(cropped_table_dir).joinpath(
                    f"{pure_posix_path.stem}_{i}.png"
                ),
                table_crop_height if rotated else table_crop_width,
                table_crop_width if rotated else table_crop_height,
                "custom-cropped-tables",
            )
            for label, box in zip(detection_labels.tolist(), detection_boxes.unbind()):
                if annotations.iob(box, table_crop_box) >= 0.5:
                    relative_box = box - table_crop_box[:2].repeat(2)
                    element = annotations.create_pascal_voc_object_element(
                        detection_id2label[label],
                        (
                            annotations.rotate90_box_with_original_width(
                                relative_box, table_crop_width
                            )
                            if rotated
                            else relative_box
                        ).numpy(),
                    )
                    annotation.append(element)
            cropped_pascal_voc_xml_output_path = os.path.join(
                output_table_cropped_pascal_voc_xml_dir,
                f"{pure_posix_path.stem}_{i}_cropped.xml",
            )
            print(cropped_pascal_voc_xml_output_path)
            annotations.save_xml_pascal_voc(
                annotation, cropped_pascal_voc_xml_output_path
            )
    # detection_labels[detection_table_indicator]

    if not cropped_table_images:
        print(f"No tables found in {pure_posix_path}.")
        return

    def compute_sresults():
        print("Computing sresults")
        encodings = feature_extractor(cropped_table_images, return_tensors="pt")
        with torch.no_grad():
            soutputs = smodel(**encodings)
        del encodings

        target_sizes = [
            cropped_table_image.size[::-1]
            for cropped_table_image in cropped_table_images
        ]
        sresults = feature_extractor.post_process_object_detection(
            soutputs, threshold=0.7, target_sizes=target_sizes
        )
        del soutputs
        assert len(sresults) == len(cropped_table_images)
        return sresults

    sresults = functools.cache(compute_sresults)

    n = len(smodel.config.id2label)

    if output_annotated_structure_image_dir:
        os.makedirs(output_annotated_structure_image_dir, exist_ok=True)
        for i, (cropped_table_image, sresult) in enumerate(
            zip(cropped_table_images, sresults())
        ):
            _, a, b = annotations.optimal_width_height(cropped_table_image.size, n)
            _, axs = plt.subplots(b, a, figsize=(12, 12))
            axes = axs.flatten()
            for ax in axes:
                ax.set_axis_off()
                ax.imshow(cropped_table_image)
            for (label_id, label_string), ax in zip(
                smodel.config.id2label.items(), axes
            ):
                label_indicator = sresult["labels"] == label_id
                plot_results(
                    ax,
                    sresult["scores"][label_indicator],
                    [label_string] * label_indicator.sum(),
                    sresult["boxes"][label_indicator],
                )

        plt.gcf().set_size_inches(
            20 * a * cropped_table_image.size[0] / b / cropped_table_image.size[1], 20
        )
        annotated_structure_png = os.path.join(
            output_annotated_structure_image_dir,
            f"{pure_posix_path.stem}_{i}_structure.png",
        )
        print(annotated_structure_png)
        plt.savefig(annotated_structure_png, bbox_inches="tight")
        plt.close()

    if output_structure_pascal_voc_xml_dir:
        os.makedirs(output_structure_pascal_voc_xml_dir, exist_ok=True)
        for i, (cropped_table_image, sresult) in enumerate(
            zip(cropped_table_images, sresults())
        ):
            annotation = annotations.create_pascal_voc_page_element(
                pathlib.PurePosixPath(cropped_table_dir).joinpath(
                    f"{pure_posix_path.stem}_{i}.png"
                ),
                cropped_table_image.size[0],
                cropped_table_image.size[1],
                "custom-structure",
            )
            for label, box in zip(
                sresult["labels"].tolist(), sresult["boxes"].unbind()
            ):
                element = annotations.create_pascal_voc_object_element(
                    smodel.config.id2label[label], box.numpy()
                )
                annotation.append(element)
            dest_annot_path = os.path.join(
                output_structure_pascal_voc_xml_dir,
                f"{pure_posix_path.stem}_{i}_structure.xml",
            )
            print(dest_annot_path)
            annotations.save_xml_pascal_voc(annotation, dest_annot_path)

    def compute_concat_sresults():
        return {
            "whole_image_structure_label_values": torch.cat(
                tuple(
                    to_whole_image_structure_label_tensor(
                        sresult["labels"], rotated, smodel.config.id2label
                    )
                    for rotated, sresult in zip(rotations, sresults(), strict=True)
                )
            ),
            "scores": torch.cat(tuple(sresult["scores"] for sresult in sresults())),
            "whole_image_structure_boxes": torch.cat(
                tuple(
                    (
                        annotations.rotate270_boxes_with_original_width(
                            sresult["boxes"], table_crop_box[3] - table_crop_box[1]
                        )
                        if rotated
                        else sresult["boxes"]
                    )
                    + torch.unsqueeze(table_crop_box[:2].repeat(2), dim=0)
                    for table_crop_box, rotated, sresult in zip(
                        table_crop_boxes, rotations, sresults(), strict=True
                    )
                )
            ),
        }

    concat_sresults = functools.cache(compute_concat_sresults)

    if use_detection_tables_in_whole_image_structure:
        stable_indicator = torch.logical_or(
            concat_sresults()["whole_image_structure_label_values"]
            == WholeImageStructureLabel.TABLE.value,
            concat_sresults()["whole_image_structure_label_values"]
            == WholeImageStructureLabel.TABLE_ROTATED.value,
        )
        if detection_table_indicator.sum() != stable_indicator.sum():
            raise ValueError(
                "WARNING Cannot use detection table: {} {}".format(
                    detection_table_indicator, stable_indicator
                )
            )
        concat_sresults()["whole_image_structure_boxes"][
            stable_indicator
        ] = detection_boxes[detection_table_indicator]
        concat_sresults()["scores"][stable_indicator] = detection_scores[
            detection_table_indicator
        ]

    if output_whole_image_structure_pascal_voc_xml_dir:
        os.makedirs(output_whole_image_structure_pascal_voc_xml_dir, exist_ok=True)
        annotation = annotations.create_pascal_voc_page_element(
            pure_posix_path,
            rgb_image.size[0],
            rgb_image.size[1],
            "custom-whole-structure",
        )
        for label, box in zip(
            concat_sresults()["whole_image_structure_label_values"].tolist(),
            concat_sresults()["whole_image_structure_boxes"].unbind(),
        ):
            element = annotations.create_pascal_voc_object_element(
                WholeImageStructureLabel(label).external(), box.numpy()
            )
            annotation.append(element)
        # find structure/data/semimanual_whole_image_pascal_voc_xml -type f |
while read f; do
mv $f ${f%%_whole_structure.xml}.xml;
done
        dest_annot_path = os.path.join(
            output_whole_image_structure_pascal_voc_xml_dir,
            f"{pure_posix_path.stem}.xml",
        )
        print(dest_annot_path)
        annotations.save_xml_pascal_voc(annotation, dest_annot_path)

    if output_whole_image_structure_pascal_voc_xml_dir:
        whole_image_structure_labels = concat_sresults()[
            "whole_image_structure_label_values"
        ]
        whole_image_structure_boxes = concat_sresults()["whole_image_structure_boxes"]
        whole_image_structure_scores = concat_sresults()["scores"]
    del concat_sresults

    if input_whole_image_structure_pascal_voc_xml_dir:
        in_annot_path = os.path.join(
            input_whole_image_structure_pascal_voc_xml_dir,
            f"{pure_posix_path.stem}.xml",
        )
        input_boxes, input_labels = annotations.read_pascal_voc(in_annot_path)
        input_whole_image_structure_boxes = torch.as_tensor(
            input_boxes, dtype=torch.float32
        ).reshape(len(input_boxes), 4)
        input_whole_image_structure_labels = torch.as_tensor(
            [
                WholeImageStructureLabel.from_external(input_label).value
                for input_label in input_labels
            ],
            dtype=torch.int64,
        )
        if (
            input_whole_image_structure_pascal_voc_xml_dir
            == output_whole_image_structure_pascal_voc_xml_dir
        ):
            assert torch.equal(
                input_whole_image_structure_labels, whole_image_structure_labels
            ), (
                input_whole_image_structure_labels,
                whole_image_structure_labels,
            )
            assert torch.equal(
                input_whole_image_structure_boxes, whole_image_structure_boxes
            ), (
                input_whole_image_structure_boxes,
                whole_image_structure_boxes,
            )
        else:
            (
                whole_image_structure_labels,
                whole_image_structure_boxes,
                whole_image_structure_scores,
            ) = (
                input_whole_image_structure_labels,
                input_whole_image_structure_boxes,
                torch.ones_like(
                    input_whole_image_structure_labels, dtype=torch.float32
                ),
            )
        del input_whole_image_structure_labels, input_whole_image_structure_boxes

    if output_whole_image_sanitized_structure_pascal_voc_xml_dir:
        os.makedirs(
            output_whole_image_sanitized_structure_pascal_voc_xml_dir, exist_ok=True
        )
        sanitized_whole_image_structure_boxes = whole_image_structure_boxes.clone()
        indices = sorted(
            range(len(whole_image_structure_boxes)),
            key=lambda index: whole_image_structure_labels[index].item(),
        )
        # TODO: Keep all inside tables, Make projected row equal to a row, make table column header equal to a row.
        for j, j_index in enumerate(indices):
            for i in range(0, j):
                trust_first_adjust_second(
                    sanitized_whole_image_structure_boxes[indices[i]],
                    sanitized_whole_image_structure_boxes[j_index],
                )
        print(
            "Adjustments: {}".format(
                (
                    sanitized_whole_image_structure_boxes != whole_image_structure_boxes
                ).sum()
            )
        )
        annotation = annotations.create_pascal_voc_page_element(
            pure_posix_path,
            rgb_image.size[0],
            rgb_image.size[1],
            "custom-whole-structure-adjusted",
        )
        for label, box in zip(
            whole_image_structure_labels,
            sanitized_whole_image_structure_boxes,
        ):
            element = annotations.create_pascal_voc_object_element(
                WholeImageStructureLabel(label.item()).external(), box.numpy()
            )
            annotation.append(element)
        # find structure/data/semimanual_whole_image_pascal_voc_xml -type f |
while read f; do
mv $f ${f%%_whole_structure.xml}.xml;
done
        dest_annot_path = os.path.join(
            output_whole_image_sanitized_structure_pascal_voc_xml_dir,
            f"{pure_posix_path.stem}.xml",
        )
        print(dest_annot_path)
        annotations.save_xml_pascal_voc(annotation, dest_annot_path)

    # TODO: output_whole_image_crop_structure_pascal_voc_xml_dir

    if output_whole_image_annotated_structure_image_dir:
        os.makedirs(output_whole_image_annotated_structure_image_dir, exist_ok=True)
        _, a, b = annotations.optimal_width_height(
            rgb_image.size, len(WholeImageStructureLabel)
        )
        _, axs = plt.subplots(b, a, figsize=(24, 24))
        axes = axs.flatten()
        for ax in axes:
            ax.set_axis_off()
        for label, ax in zip(WholeImageStructureLabel, axes):
            label_indicator = whole_image_structure_labels == label.value
            if label_indicator.sum():
                ax.imshow(rgb_image)
                plot_results(
                    ax,
                    whole_image_structure_scores[label_indicator],
                    [label.external()] * label_indicator.sum(),
                    whole_image_structure_boxes[label_indicator],
                )
        plt.gcf().set_size_inches(
            40 * a * rgb_image.size[0] / b / rgb_image.size[1], 40
        )
        annotated_whole_image_structure_png = os.path.join(
            output_whole_image_annotated_structure_image_dir,
            f"{pure_posix_path.stem}_{i}_whole_image_structure.png",
        )
        print(annotated_whole_image_structure_png)
        plt.savefig(annotated_whole_image_structure_png, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process tables")
    parser.add_argument(
        "--page_images", nargs="+", help="Comma-separated list of images."
    )
    parser.add_argument(
        "--annotated_detection_image_dir",
        help="Where to output annotated detection images.",
    )
    parser.add_argument(
        "--output_detection_pascal_voc_xml_dir",
        help="Where to output pascal voc XML files.",
    )
    parser.add_argument(
        "--input_detection_pascal_voc_xml_dir",
        help="Where to read pascal voc XML files from.",
    )
    parser.add_argument(
        "--distance_to_tables_for_line_removal",
        type=int,
        help="How many pixels around tables to remove lines.",
    )
    parser.add_argument(
        "--line_width",
        type=int,
        help="Width of line.",
    )
    parser.add_argument(
        "--min_removed_lines",
        type=int,
        help="How many removed lines to require for writing a new image.",
    )
    parser.add_argument(
        "--reprocess_removed_lines",
        action="store_true",
        help="If true, process an image even if the output exists and is valid.",
    )
    parser.add_argument(
        "--output_image_without_lines_dir",
        help="Where to output the image with lines in tables removed.",
    )
    parser.add_argument(
        "--output_image_with_contours_dir",
        help="Where to output the image contours in tables removed.",
    )
    parser.add_argument(
        "--horizontal_structuring_kernel_length",
        type=int,
        help="If missing then height // H_STRUCTURE_ELEMENT_DIMENSION_SCALE",
    )
    parser.add_argument(
        "--vertical_structuring_kernel_length",
        type=int,
        help="If missing then width // V_STRUCTURE_ELEMENT_DIMENSION_SCALE",
    )
    parser.add_argument(
        "--line_colors",
        nargs="*",
        help="Comma-separated list of colors. When empty distinct colors will be generated.",
        default=["(1, 1, 1)"],
    )
    parser.add_argument("--structure_table_padding", type=int, default=40)
    parser.add_argument(
        "--cropped_table_dir",
        help="Where to output the cropped detection table images.",
    )
    parser.add_argument(
        "--output_table_cropped_pascal_voc_xml_dir",
        help="Where to output cropped pascal voc XML files for each table.",
    )
    parser.add_argument(
        "--output_annotated_structure_image_dir",
        help="Where to output annotated structure images for each table.",
    )
    parser.add_argument(
        "--output_structure_pascal_voc_xml_dir",
        help="Where to output structure pascal voc XML files for each table.",
    )
    parser.add_argument(
        "--output_whole_image_sanitized_structure_pascal_voc_xml_dir",
        help="Where to output sanitized the whole image structures.",
    )
    parser.add_argument(
        "--use_detection_tables_in_whole_image_structure",
        action=argparse.BooleanOptionalAction,
        help="Whether to override the structure tables with detection tables.",
    )
    parser.add_argument(
        "--output_whole_image_structure_pascal_voc_xml_dir",
        help="Where to output whole image structure pascal voc XML files for each table.",
    )
    parser.add_argument(
        "--input_whole_image_structure_pascal_voc_xml_dir",
        help="Where to read full-image structure pascal voc XML files from.",
    )
    parser.add_argument(
        "--output_whole_image_annotated_structure_image_dir",
        help="Where to output whole image annotated structure images.",
    )
    parser.add_argument(
        "--print0",
        action="store_true",
        help="If true, print a null character after processing each file.",
    )
    args = parser.parse_args()

    feature_extractor = DetrImageProcessor()
    detection_model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-detection"
    )
    smodel = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-structure-recognition"
    )

    print("Images: {}".format(len(args.page_images)), flush=True)
    for page_image in args.page_images:
        main(
            feature_extractor,
            detection_model,
            smodel,
            page_image,
            args.annotated_detection_image_dir,
            args.output_detection_pascal_voc_xml_dir,
            args.input_detection_pascal_voc_xml_dir,
            args.distance_to_tables_for_line_removal,
            args.line_width,
            args.min_removed_lines,
            args.reprocess_removed_lines,
            args.output_image_without_lines_dir,
            args.output_image_with_contours_dir,
            args.horizontal_structuring_kernel_length,
            args.vertical_structuring_kernel_length,
            [
                ast.literal_eval(line_color)
                for line_color in args.line_colors
                if line_color
            ],
            args.structure_table_padding,
            args.cropped_table_dir,
            args.output_table_cropped_pascal_voc_xml_dir,
            args.output_annotated_structure_image_dir,
            args.output_structure_pascal_voc_xml_dir,
            args.output_whole_image_sanitized_structure_pascal_voc_xml_dir,
            args.use_detection_tables_in_whole_image_structure,
            args.output_whole_image_structure_pascal_voc_xml_dir,
            args.input_whole_image_structure_pascal_voc_xml_dir,
            args.output_whole_image_annotated_structure_image_dir,
        )
        if args.print0:
            print("\x00", flush=True)
