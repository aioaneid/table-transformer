import argparse
import pathlib
import re
import collections
import pprint
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt


def get_args_parser():
    parser = argparse.ArgumentParser("Performance plots", add_help=False)
    parser.add_argument("--input_logs", type=pathlib.Path, nargs="*")
    parser.add_argument("--file_regex", type=str)
    parser.add_argument("--output_dir", type=pathlib.Path)
    parser.add_argument("--aspect", type=float)
    parser.add_argument("--markers", type=str, nargs="*")
    parser.add_argument("--default_linestyles", type=str, nargs="+")
    parser.add_argument(
        "--vertical_axis_labels",
        action=argparse.BooleanOptionalAction,
        help="Show labels on vertical axis.",
        default=True
    )
    return parser


_ALL_BOXES = ""  # " (all boxes)"
_EQUIV = " (eq.)"


def draw_plots():
    args = parser.parse_args()
    print(args)
    filename_pattern = re.compile(args.file_regex)
    print("filename_pattern: {}".format(filename_pattern))
    metrics_pattern = re.compile(
        "pubmed: AP50: (?P<ap50>[0-9.]+), AP75: (?P<ap75>[0-9.]+), AP: (?P<ap>[0-9.]+), AR: (?P<ar>[0-9.]+)"
    )
    print("metrics_pattern: {}".format(metrics_pattern))
    train_len_pattern = re.compile(r"train_len: (?P<tl>\d+)")
    epoch_pattern = re.compile(r"^Epoch:\s*\[(?P<ei>\d+)\]")

    d = collections.defaultdict(list)
    for filepath in args.input_logs:
        print(filepath)
        filename_match = filename_pattern.match(filepath.name)
        has_fraction_group = "fraction" in filename_pattern.groupindex
        has_all_object_segmented_group = (
            "all_objects_segmented" in filename_pattern.groupindex
        )
        has_kept_group = "kept" in filename_pattern.groupindex
        if has_fraction_group:
            fraction_group = filename_match.group("fraction") if filename_match else None
            fraction = "{:.2%}".format(float(fraction_group)) if fraction_group else "unknown"
        epoch_group = filename_match.group("epoch") if filename_match else None
        epoch_from_filename = int(epoch_group) if epoch_group else None
        assert epoch_from_filename is None or epoch_from_filename >= 1
        values = None
        latest_epoch = None
        file_stats = {}
        with open(filepath, mode="rt") as f:
            for line in f:
                epoch_match = epoch_pattern.match(line)
                if epoch_match:
                    epoch_from_file_content = int(epoch_match.group("ei")) + 1
                    if epoch_from_filename:
                        assert epoch_from_file_content == epoch_from_filename
                    latest_epoch = epoch_from_file_content
                # If "fraction" group is missing then read train_len from "train_len: 50000 batch_size: 2 args.fused: None".
                train_len_match = train_len_pattern.match(line)
                if train_len_match and not has_fraction_group:
                    # fraction = "images: {}".format(train_len_match.group("tl"))
                    fraction = "all"
                metrics_match = metrics_pattern.match(line)
                if metrics_match:
                    assert latest_epoch
                    values = {
                        key: float(value)
                        for key, value in metrics_match.groupdict().items()
                    }
                    file_stats[latest_epoch] = values
        for epoch, values in file_stats.items():
            v = d[
                "{}{}{}".format(
                    "{}{}".format(
                        filename_match.group("kept"),
                        ""
                        if not has_all_object_segmented_group
                        else _EQUIV
                        if filename_match.group("all_objects_segmented")
                        else " / img",
                    )
                    if has_kept_group and filename_match.group("kept")
                    else "",
                    "" if has_kept_group and filename_match.group("kept") else fraction,
                    ""
                    if not has_all_object_segmented_group
                    else _ALL_BOXES
                    if filename_match.group("all_objects_segmented")
                    else ""
                    if has_kept_group and filename_match.group("kept")
                    else " (random box subset)",
                )
            ]
            v.extend([None] * (epoch - len(v)))
            v[epoch - 1] = (filepath, values)
    print(pprint.pformat(d))

    epochs = max((len(v) for v in d.values()))
    # markers = ['.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
    markers = [
        "1",
        "o",
        "+",
        "X",
        "*",
        ">",
        "x",
        "P",
        "^",
        "<",
        "h",
        "s",
        "p",
        "H",
        "D",
        "8",
        "d",
        "v",
    ]
    # markers = [m for m, func in Line2D.markers.items() if func != 'nothing' and m not in Line2D.filled_markers]
    number_regex = re.compile(r"\d+")
    for key in ("ap50", "ap75", "ap", "ar"):
        fig = plt.figure(figsize=(8, 8 / args.aspect) if args.aspect else None)
        ax = fig.subplots()  # add_subplot(111, aspect="equal")
        space_for_legend = 4.4 if len(d) > 6 else 0
        ax.set_xlim(0.75, epochs + 0.25 + space_for_legend)
        ax.set_xticks(list(range(1, epochs + 1)))
        if not args.vertical_axis_labels:
            ax.get_yaxis().set_ticks([])
        # ax.set_ylim(0, 1)
        # s = math.ceil(math.sqrt(len(d)))
        equiv_linestyle = (0, (2, 1, 1, 1))
        for index, (fraction, v) in enumerate(d.items()):
            x = [i + 1 for i in range(epochs) if i < len(v) and v[i] is not None]
            y = [value[key] for _, value in v if value is not None]
            number_match = number_regex.search(fraction)
            objects_per_image = int(number_match.group()) if number_match else 100
            marker = (
                args.markers[index % len(args.markers)]
                if args.markers
                else markers[objects_per_image % len(markers)]
            )
            print(fraction, marker)
            ax.plot(
                x,
                y,
                label=fraction,
                # linestyle=(0, (1 + index // s, 1 + index % s)),
                linestyle="solid"
                if fraction.startswith("images: ") or fraction == "all"
                else equiv_linestyle
                if _EQUIV in fraction
                else (args.default_linestyles[index % len(args.default_linestyles)]),
                # markersize=5,
                marker=marker,
                fillstyle="none",
            )
            ax.set(xlabel="Epoch", ylabel=key.upper())
        ax.legend(loc="lower right")
        # ax.set_title(key.upper())
        # fig.legend()
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            image_path = pathlib.Path(args.output_dir, "{}_fractions.svg".format(key))
            print(image_path)
            fig.savefig(
                image_path,
                # bbox_extra_artists=(lgd,),
                bbox_inches="tight",
            )
            plt.close()


if __name__ == "__main__":
    parser = get_args_parser()
    draw_plots()
