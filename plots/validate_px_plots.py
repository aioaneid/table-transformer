import argparse
import pathlib
import re
import collections
import typing
import numpy as np
import seaborn as sns
import colorcet as cc
import pprint

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def get_args_parser():
    parser = argparse.ArgumentParser("Px validation plots", add_help=False)
    parser.add_argument("--standard_logs", type=pathlib.Path, nargs="*", default=[])
    parser.add_argument(
        "--standard_log_pattern",
        type=re.compile,
        default=r".*_test_split_name_(?P<test_split_name>.*)_epoch_(?P<epoch>\d+)[.]log",
    )
    parser.add_argument(
        "--enable_bounds_logs", type=pathlib.Path, nargs="*", default=[]
    )
    parser.add_argument(
        "--enable_bounds_log_pattern",
        type=re.compile,
        default=r".*_test_split_name_(?P<test_split_name>.*)_epoch_(?P<epoch>\d+)[.]log",
    )
    parser.add_argument("--px_logs", type=pathlib.Path, nargs="*", default=[])
    parser.add_argument(
        "--px_log_pattern",
        type=re.compile,
        default=r".*-px-(?P<px>[0-9.]+)-.*_test_split_name_(?P<test_split_name>.*)_epoch_(?P<epoch>\d+)[.]log",
    )
    parser.add_argument("--pxs_logs", type=pathlib.Path, nargs="*", default=[])
    parser.add_argument(
        "--pxs_log_pattern",
        type=re.compile,
        default=r".*-pxs-(?P<px>[0-9.]+)-.*_test_split_name_(?P<test_split_name>.*)_epoch_(?P<epoch>\d+)[.]log",
    )
    parser.add_argument("--pxt_logs", type=pathlib.Path, nargs="*", default=[])
    parser.add_argument(
        "--pxt_log_pattern",
        type=re.compile,
        default=r".*-pxt-(?P<px>[0-9.]+)-.*_test_split_name_(?P<test_split_name>.*)_epoch_(?P<epoch>\d+)[.]log",
    )
    parser.add_argument("--pxn_logs", type=pathlib.Path, nargs="*", default=[])
    parser.add_argument(
        "--pxn_log_pattern",
        type=re.compile,
        default=r".*-pxn-(?P<px>[0-9.]+)-.*_test_split_name_(?P<test_split_name>.*)_epoch_(?P<epoch>\d+)[.]log",
    )
    parser.add_argument("--suf_logs", type=pathlib.Path, nargs="*", default=[])
    parser.add_argument(
        "--suf_log_pattern",
        type=re.compile,
        default=r".*-suf-(?P<suf>.*)(?:_kept_(?P<kept>[0-9.]+))?-eb-(?P<eb>.*enable_bounds)-.*_test_split_name_(?P<test_split_name>.*)_epoch_(?P<epoch>\d+)[.]log",
    )
    # /home/mluser/work/tmp/detection/validate/
    # model_code-table-transformer-suf-single-eb-no-enable_bounds-output-train_instant-202403080000000000/
    # pubmed_mode_validate_code_table-transformer_test_split_name_val_epoch_1.log
    parser.add_argument("--asym_logs", type=pathlib.Path, nargs="*", default=[])
    parser.add_argument(
        "--asym_log_pattern",
        type=re.compile,
        default=r".*-dashed-(?P<dash>[0-9.]+-[0-9.]+-[0-9.]+-[0-9.]+)-(?P<expansion>.*)-(?P<px>[0-9.]+)-.*_test_split_name_(?P<test_split_name>.*)_epoch_(?P<epoch>\d+)[.]log",
    )
    parser.add_argument("--upx_logs", type=pathlib.Path, nargs="*", default=[])
    parser.add_argument(
        "--upx_log_pattern",
        type=re.compile,
        default=r".*-upx-(?P<upx>[0-9.]+)-px-(?P<px>[0-9.]+)-.*_test_split_name_(?P<test_split_name>.*)_epoch_(?P<epoch>\d+)[.]log",
    )
    parser.add_argument("--umidpix_logs", type=pathlib.Path, nargs="*", default=[])
    parser.add_argument(
        "--umidpix_log_pattern",
        type=re.compile,
        default=r".*-upx-(?P<upx>[0-9.]+)-midpix-(?P<px>[0-9.]+)-.*_test_split_name_(?P<test_split_name>.*)_epoch_(?P<epoch>\d+)[.]log",
    )
    parser.add_argument("--output_dir", type=pathlib.Path)
    parser.add_argument("--aspect", type=float, default=2)
    parser.add_argument(
        "--markers",
        type=str,
        nargs="*",
        default=[
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
        ],
    )
    parser.add_argument(
        "--default_linestyles",
        type=str,
        nargs="*",
        default=["solid", "dashed", "dashed", "dashed", "dashed", "dashed"],
    )
    parser.add_argument(
        "--vertical_axis_labels",
        action=argparse.BooleanOptionalAction,
        help="Show labels on vertical axis.",
        default=True,
    )
    parser.add_argument(
        "--annotate",
        action=argparse.BooleanOptionalAction,
        help="Show text annotations.",
        default=True,
    )
    parser.add_argument(
        "--min_epochs",
        type=int,
        default=0,
    )
    return parser


class CocoPerf(typing.NamedTuple):
    ap50: float
    ap75: float
    ap: float
    ar: float


class AveragedStats(typing.NamedTuple):
    class_error_unscaled: float
    cardinality_error_unscaled: float


metrics_pattern = re.compile(
    r"pubmed: AP50: (?P<ap50>[0-9.]+), AP75: (?P<ap75>[0-9.]+), AP: (?P<ap>[0-9.]+), AR: (?P<ar>[0-9.]+)"
)
averaged_stats_pattern = re.compile(
    r"Averaged stats: .* class_error_unscaled: [0-9.]+ \((?P<class_error_unscaled>[0-9.]+)\) .* cardinality_error_unscaled: [0-9.]+ \((?P<cardinality_error_unscaled>[0-9.]+)\)"
)
patterns = (metrics_pattern, averaged_stats_pattern)


def read_coco_performance(input_file):
    matches = [None] * len(patterns)
    for line in reversed(list(input_file)):
        for i, pattern in enumerate(patterns):
            if not matches[i]:
                matches[i] = pattern.match(line)
    return (
        CocoPerf(**{key: float(value) for key, value in matches[0].groupdict().items()})
        if matches[0]
        else None
    ), (
        AveragedStats(
            **{key: float(value) for key, value in matches[1].groupdict().items()}
        )
        if matches[1]
        else None
    )


def update_list(index, value, l):
    l.extend([(None, None)] * max(index + 1 - len(l), 0))
    l[index] = value


def draw_plots():
    args = parser.parse_args()
    d = collections.defaultdict(list)
    for px_type, (log_pattern, logs) in {
        "std": (args.standard_log_pattern, args.standard_logs),
        "eb": (args.enable_bounds_log_pattern, args.enable_bounds_logs),
        "px": (args.px_log_pattern, args.px_logs),
        "pxs": (args.pxs_log_pattern, args.pxs_logs),
        "pxt": (args.pxt_log_pattern, args.pxt_logs),
        "pxn": (args.pxn_log_pattern, args.pxn_logs),
    }.items():
        for file_path in logs:
            if not file_path.exists():
                continue
            print(pprint.pformat((px_type, file_path)))
            match = log_pattern.match(str(file_path))
            epoch = int(match.group("epoch"))
            px = float(match.group("px")) if "px" in log_pattern.groupindex else 0
            with open(file_path, mode="rt") as f:
                coco_perf = read_coco_performance(f)
            update_list(
                epoch - 1,
                coco_perf,
                d[
                    "{} {:<3}: {: 3.2f}".format(
                        match.group("test_split_name"), px_type, px
                    )
                ],
            )
    for file_path in args.suf_logs:
        if not file_path.exists():
            continue
        print(pprint.pformat(("suf", file_path)))
        match = args.suf_log_pattern.match(str(file_path))
        epoch = int(match.group("epoch"))
        kept = (
            float(match.group("kept") or "inf")
            if "kept" in log_pattern.groupindex
            else 0
        )
        eb = match.group("eb").replace("enable_bounds", "eb")
        suf = match.group("suf")
        with open(file_path, mode="rt") as f:
            coco_perf = read_coco_performance(f)
        update_list(
            epoch - 1,
            coco_perf,
            d[
                "{} {:<3} ({}): {: 3.2f}".format(
                    match.group("test_split_name"), suf, eb, kept
                )
            ],
        )

    for file_path in args.asym_logs:
        if not file_path.exists():
            continue
        print(pprint.pformat(("asym", file_path)))
        match = args.asym_log_pattern.match(str(file_path))
        epoch = int(match.group("epoch"))
        expansion = match.group("expansion")
        px = float(match.group("px"))
        dash = match.group("dash")
        with open(file_path, mode="rt") as f:
            coco_perf = read_coco_performance(f)
        update_list(
            epoch - 1,
            coco_perf,
            d[
                "{} {:<3}-{}-{:02.0f}".format(
                    match.group("test_split_name"), dash, expansion, px
                )
            ],
        )

    for logs_type, file_paths, log_pattern in [
        ("upx", args.upx_logs, args.upx_log_pattern),
        ("umidpix", args.umidpix_logs, args.umidpix_log_pattern),
    ]:
        for file_path in file_paths:
            if not file_path.exists():
                continue
            print(pprint.pformat((logs_type, file_path)))
            match = log_pattern.match(str(file_path))
            assert match, file_path
            epoch = int(match.group("epoch"))
            upx = float(match.group("upx"))
            px = float(match.group("px"))
            with open(file_path, mode="rt") as f:
                coco_perf = read_coco_performance(f)
            update_list(
                epoch - 1,
                coco_perf,
                d[
                    "{} {}-{:02.0f}-px-{:02.0f}".format(
                        match.group("test_split_name"), logs_type, upx, px
                    )
                ],
            )

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    global metrics_pattern
    epochs = max((len(value) for value in d.values()), default=0)
    for pattern_index, (pattern, extreme_fn, extreme_offset, legend_loc) in enumerate(
        zip(patterns, (np.max, np.min), (-0.1, 0.1), ("lower center", "upper center"))
    ):
        for key in pattern.groupindex:
            image_path = (
                pathlib.Path(args.output_dir, "{}_px_coco_metrics.svg".format(key))
                if args.output_dir
                else None
            )
            print(image_path)
            fig = plt.figure(figsize=(8, 8 / args.aspect) if args.aspect else None)
            ax = fig.subplots()  # add_subplot(111, aspect="equal")
            space_for_legend = 4.4 if len(d) > 6 else 0
            ax.set_xlim(args.min_epochs + 0.75, epochs + 0.25 + space_for_legend)
            ax.set_xticks(np.arange(max(epochs, 20), args.min_epochs, -1), minor=True)
            ax.set_xticks(
                np.arange((max(epochs, 20) + 4) // 5 * 5, args.min_epochs, -5)
            )

            ax.tick_params(top=True, right=True, labelleft=False, labelright=True)
            ax.tick_params(
                top=True, right=True, labelleft=False, labelright=True, which="minor"
            )
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            if not args.vertical_axis_labels:
                ax.get_yaxis().set_ticks([])
            for index, ((label, v), c) in enumerate(
                zip(
                    sorted(
                        d.items(),
                        key=lambda x: (
                            "std" not in x[0],
                            "-" in x[0],
                            x[0],
                        ),
                    ),
                    sns.color_palette(cc.glasbey, n_colors=len(d)),
                )
            ):
                x = [
                    i + 1
                    for i in range(args.min_epochs, epochs)
                    if i < len(v) and v[i][pattern_index]
                ]
                if not x:
                    continue
                y = [
                    getattr(v[i][pattern_index], key)
                    for i in range(args.min_epochs, epochs)
                    if i < len(v) and v[i][pattern_index]
                ]
                marker = args.markers[index % len(args.markers)]
                ax.plot(
                    x,
                    y,
                    label=label,
                    # linestyle=(0, (1 + index // s, 1 + index % s)),
                    linestyle=args.default_linestyles[
                        index % len(args.default_linestyles)
                    ],
                    markersize=4,
                    marker=marker,
                    color=c,
                    fillstyle="none",
                    alpha=0.85,
                    linewidth=0.875,
                )
                if args.annotate:
                    (ks,) = np.nonzero(y == extreme_fn(y))
                    for k in ks:
                        ax.annotate(
                            "{:.4f}".format(y[k]),
                            xy=(x[k], y[k]),
                            xytext=(x[k], y[k] + (max(y, default=0) - min(y, default=0)) * extreme_offset * (2 + (index * 3 + k * 7) % 5)),
                            arrowprops=dict(arrowstyle="->", color=c),
                            color=c,
                            alpha=0.85,
                        )
            ax.set(xlabel="Epoch", ylabel=key.upper())
            ax.legend(loc=legend_loc, fontsize="small", ncol=1)
            if args.output_dir:
                fig.savefig(
                    image_path,
                    bbox_inches="tight",
                )
            plt.close()


if __name__ == "__main__":
    parser = get_args_parser()
    draw_plots()
