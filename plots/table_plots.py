from typing import Literal
import numpy as np

import matplotlib.patches as patches


def scale_bbox(bbox, scale_x, scale_y):
    bbox = np.array(bbox)
    bbox[0::2] *= scale_x
    bbox[1::2] *= scale_y
    return bbox


def shrink_bbox_in_place(bbox, padding):
    bbox[0] += padding
    bbox[1] += padding
    bbox[2] -= padding
    bbox[3] -= padding
    if bbox[0] > bbox[2]:
        bbox[0] = bbox[2] = (bbox[0] + bbox[2]) / 2
    if bbox[1] > bbox[3]:
        bbox[1] = bbox[3] = (bbox[1] + bbox[3]) / 2


_HEADER_COLOR = "darkorange"
_HEADER_HATCH = "."
_PROJECTED_ROW_HEADER_COLOR = "darkturquoise"
_PROJECTED_ROW_HEADER_HATCH = "//"


def get_bbox_decorations(data_type, label):
    if label == 0:
        if data_type == "detection":
            return "brown", 0.05, 3, "//"
        else:
            return "brown", 0, 3, None
    elif label == 1:
        return "red", 0.1, 2, None
    elif label == 2:
        return "blue", 0.1, 2, None
    elif label == 3:
        return _HEADER_COLOR, 0.2, 3, _HEADER_HATCH
    elif label == 4:
        return _PROJECTED_ROW_HEADER_COLOR, 0.2, 4, _PROJECTED_ROW_HEADER_HATCH
    elif label == 5:
        return "green", 0.2, 8, "\\\\"

    return "gray", 0, 0, None


def get_rectangles(
    pred_bboxes,
    pred_labels,
    pred_scores,
    data_type: Literal["detection", "structure"],
    padding: int,
):
    rectangles = []
    for bbox, label, score in zip(pred_bboxes, pred_labels, pred_scores):
        if (
            (data_type == "structure" and not label > 5)
            or (data_type == "detection" and not label > 1)
            and score > 0.5
        ):
            color, alpha, linewidth, hatch = get_bbox_decorations(data_type, label)
            shrink_bbox_in_place(bbox, padding)
            # Fill
            rect = patches.Rectangle(
                bbox[:2],
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                linewidth=linewidth,
                alpha=alpha,
                edgecolor="none",
                facecolor=color,
                linestyle=None,
            )
            rectangles.append(rect)
            # Hatch
            rect = patches.Rectangle(
                bbox[:2],
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                linewidth=1,
                alpha=0.4,
                edgecolor=color,
                facecolor="none",
                linestyle="--",
                hatch=hatch,
            )
            rectangles.append(rect)
            # Edge
            rect = patches.Rectangle(
                bbox[:2],
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                linewidth=linewidth,
                edgecolor=color,
                facecolor="none",
                linestyle="--",
            )
            rectangles.append(rect)
    return rectangles


def get_cell_rectangles(cell, scale_x, scale_y, padding):
    rectangles = []
    bbox = cell["bbox"]
    bbox = np.array(bbox)
    bbox[0::2] *= scale_x
    bbox[1::2] *= scale_y
    shrink_bbox_in_place(bbox, padding)
    if cell["header"]:
        alpha = 0.3
    else:
        alpha = 0.08  # 0.125
    lw = 2
    rect = patches.Rectangle(
        bbox[:2],
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
        linewidth=lw,
        edgecolor="none",
        facecolor=(
            _HEADER_COLOR
            if cell["header"]
            else _PROJECTED_ROW_HEADER_COLOR if cell["subheader"] else "crimson"
        ),
        alpha=alpha,
    )
    rectangles.append(rect)
    rect = patches.Rectangle(
        bbox[:2],
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
        linewidth=lw,
        edgecolor=(
            _HEADER_COLOR
            if cell["header"]
            else _PROJECTED_ROW_HEADER_COLOR if cell["subheader"] else "crimson"
        ),
        facecolor="none",
        linestyle="--",
        alpha=0.08,
        hatch=(
            _HEADER_HATCH
            if cell["header"]
            else _PROJECTED_ROW_HEADER_HATCH if cell["subheader"] else None
        ),
    )
    rectangles.append(rect)
    rect = patches.Rectangle(
        bbox[:2],
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
        linewidth=lw,
        edgecolor=(
            _HEADER_COLOR
            if cell["header"]
            else _PROJECTED_ROW_HEADER_COLOR if cell["subheader"] else "crimson"
        ),
        facecolor="none",
        linestyle="none",
        hatch=(
            _HEADER_HATCH
            if cell["header"]
            else _PROJECTED_ROW_HEADER_HATCH if cell["subheader"] else None
        ),
    )
    rectangles.append(rect)
    rect = patches.Rectangle(
        bbox[:2],
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
        linewidth=lw,
        edgecolor="crimson",  # _HEADER_COLOR if cell["header"] else _PROJECTED_ROW_HEADER_COLOR if cell["subheader"] else "crimson",
        facecolor="none",
        linestyle="--",
        fill="none",
        alpha=1,
    )
    rectangles.append(rect)
    return rectangles
