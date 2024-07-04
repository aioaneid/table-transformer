import argparse
import pathlib
import re
import collections
import itertools
import numpy as np
import seaborn as sns
import colorcet as cc
import grits_metrics

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def get_args_parser():
    parser = argparse.ArgumentParser("Px validation plots", add_help=False)
    parser.add_argument("--msft_logs", type=pathlib.Path, nargs="*", default=[])
    parser.add_argument(
        "--msft_log_pattern",
        type=re.compile,
        default=r".*_test_split_name_(?P<test_split_name>.*)_test_max_size_(?P<test_max_size>\d+)_epoch_(?P<epoch>\d+)[.]log",
    )
    parser.add_argument("--standard_logs", type=pathlib.Path, nargs="*", default=[])
    parser.add_argument(
        "--standard_log_pattern",
        type=re.compile,
        default=r".*_test_split_name_(?P<test_split_name>.*)_test_max_size_(?P<test_max_size>\d+)_epoch_(?P<epoch>\d+)[.]log",
    )
    parser.add_argument(
        "--standard_random_logs", type=pathlib.Path, nargs="*", default=[]
    )
    parser.add_argument(
        "--standard_random_log_pattern",
        type=re.compile,
        default=r".*_test_split_name_(?P<test_split_name>.*)_test_max_size_(?P<test_max_size>\d+)_seed_(?P<seed>\d+)_epoch_(?P<epoch>\d+)[.]log",
    )
    parser.add_argument("--pxc_logs", type=pathlib.Path, nargs="*", default=[])
    parser.add_argument(
        "--pxc_log_pattern",
        type=re.compile,
        default=r".*-pxc-(?P<px>[\w_\d]+)-eb-(?P<eb>.*enable_bounds)-.*_test_split_name_(?P<test_split_name>.*)_test_max_size_(?P<test_max_size>\d+)_epoch_(?P<epoch>\d+)[.]log",
    )
    parser.add_argument("--pxct_logs", type=pathlib.Path, nargs="*", default=[])
    parser.add_argument(
        "--pxct_log_pattern",
        type=re.compile,
        default=r".*-pxct-(?P<px>[\w_\d]+)-eb-(?P<eb>.*enable_bounds)-.*_test_split_name_(?P<test_split_name>.*)_test_max_size_(?P<test_max_size>\d+)_epoch_(?P<epoch>\d+)[.]log",
    )
    parser.add_argument("--pxct_random_logs", type=pathlib.Path, nargs="*", default=[])
    parser.add_argument(
        "--pxct_random_log_pattern",
        type=re.compile,
        default=r".*-pxct-(?P<px>[\w_\d]+)-eb-(?P<eb>.*enable_bounds)-.*_test_split_name_(?P<test_split_name>.*)_test_max_size_(?P<test_max_size>\d+)_seed_(?P<seed>\d+)_epoch_(?P<epoch>\d+)[.]log",
    )
    parser.add_argument("--fix_logs", type=pathlib.Path, nargs="*", default=[])
    parser.add_argument(
        "--fix_log_pattern",
        type=re.compile,
        default=r".*-fix-eps_(?P<eps>[\w_\d]+)_px_(?P<px>[\w_\d]+)-eb-(?P<eb>.*enable_bounds)-.*_test_split_name_(?P<test_split_name>.*)_test_max_size_(?P<test_max_size>\d+)_epoch_(?P<epoch>\d+)[.]log",
    )
    parser.add_argument("--fix_all_logs", type=pathlib.Path, nargs="*", default=[])
    parser.add_argument(
        "--fix_all_log_pattern",
        type=re.compile,
        default=r".*-fix-all-eb-(?P<eb>.*enable_bounds)-.*_test_split_name_(?P<test_split_name>.*)_test_max_size_(?P<test_max_size>\d+)_epoch_(?P<epoch>\d+)[.]log",
    )
    parser.add_argument("--pxst_logs", type=pathlib.Path, nargs="*", default=[])
    parser.add_argument(
        "--pxst_log_pattern",
        type=re.compile,
        default=r".*-pxst-eps_(?P<eps>[\w_\d]+)_px_(?P<px>[\w_\d]+)-eb-(?P<eb>.*enable_bounds)-.*_test_split_name_(?P<test_split_name>.*)_test_max_size_(?P<test_max_size>\d+)_epoch_(?P<epoch>\d+)[.]log",
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
        default=1,
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--text_usetex",
        action=argparse.BooleanOptionalAction,
        help="Show text annotations.",
        default=False,
    )
    parser.add_argument(
        "--split_by_table_type",
        action=argparse.BooleanOptionalAction,
        help="Show text annotations.",
        default=True,
    )
    return parser


accuracy_con_pattern = re.compile(r"\s+Accuracy_Con: (?P<value>[0-9.]+)")
grits_top_pattern = re.compile(r"\s+GriTS_Top: (?P<value>[0-9.]+)")
grits_con_pattern = re.compile(r"\s+GriTS_Con: (?P<value>[0-9.]+)")
grits_loc_pattern = re.compile(r"\s+GriTS_Loc: (?P<value>[0-9.]+)")

simple_results_pattern = re.compile(
    r"Results on simple tables \((?P<count>\d+) total\):"
)
complex_results_pattern = re.compile(
    r"Results on complex tables \((?P<count>\d+) total\):"
)
all_results_pattern = re.compile(r"Results on all tables \((?P<count>\d+) total\):")


def find_next_match(i, lines, pattern):
    match = None
    while i and not (match := pattern.match(lines[i := i - 1])):
        continue
    return i, match


def read_next_value(i, lines, pattern):
    i, match = find_next_match(i, lines, pattern)
    return i, (float(match.group("value")) if match else None)


def read_grits(i, lines):
    i, grits_loc_value = read_next_value(i, lines, grits_loc_pattern)
    if grits_loc_value is None:
        return i, None
    i, grits_con_value = read_next_value(i, lines, grits_con_pattern)
    if grits_con_value is None:
        return i, None
    i, grits_top_value = read_next_value(i, lines, grits_top_pattern)
    if grits_top_value is None:
        return i, None
    i, accuracy_con_value = read_next_value(i, lines, accuracy_con_pattern)
    if accuracy_con_value is None:
        return i, None
    return i, grits_metrics.Grits(
        accuracy_con=accuracy_con_value,
        grits_top=grits_top_value,
        grits_con=grits_con_value,
        grits_loc=grits_loc_value,
    )


def read_grits_performance(input_file):
    lines = list(input_file)
    i = len(lines)
    i, all_grits = read_grits(i, lines)
    if not all_grits:
        return None
    i, all_results_match = find_next_match(i, lines, all_results_pattern)
    if not all_results_match:
        return None
    i, complex_grits = read_grits(i, lines)
    if not complex_grits:
        return None
    i, complex_results_match = find_next_match(i, lines, complex_results_pattern)
    if not complex_results_match:
        return None
    i, simple_grits = read_grits(i, lines)
    if not simple_grits:
        return None
    i, simple_results_match = find_next_match(i, lines, simple_results_pattern)
    if not simple_results_match:
        return None
    return grits_metrics.GritsPerf(
        simple=grits_metrics.SubsetGrits(
            int(simple_results_match.group("count")), simple_grits
        ),
        complex=grits_metrics.SubsetGrits(
            int(complex_results_match.group("count")), complex_grits
        ),
        all=grits_metrics.SubsetGrits(int(all_results_match.group("count")), all_grits),
    )


def update_list(index, value, l):
    l.extend([None] * max(index + 1 - len(l), 0))
    l[index] = value


def update_list_test_max_size(index, test_max_size, perf, l):
    l.extend([None] * max(index + 1 - len(l), 0))
    if not l[index] or l[index][0] < test_max_size:
        l[index] = test_max_size, perf


def grits_name(key, *, text_usetex):
    if text_usetex:
        match key:
            case "accuracy_con":
                return "Acc_{Con}"
            case "grits_top":
                return "GriTS_{Top}"
            case "grits_con":
                return "GriTS_{Con}"
            case "grits_loc":
                return "GriTS_{Loc}"
    match key:
        case "accuracy_con":
            return key
        case "grits_top":
            return "GriTS Topology"
        case "grits_con":
            return "GriTS Content"
        case "grits_loc":
            return "GriTS Location"
    


def draw_line(table_type, label, v, c, epochs, index, key, args, ax):
    x = [i + 1 for i in range(args.min_epochs - 1, epochs) if i < len(v) and v[i]]
    y = [
        getattr(getattr(value[1], table_type).grits, key)
        for i, value in enumerate(v)
        if i >= args.min_epochs - 1 and i < epochs and value is not None
    ]
    test_max_size_table_count_list = [
        (value[0], getattr(value[1], table_type).table_count)
        for i, value in enumerate(v)
        if i >= args.min_epochs - 1 and i < epochs and value is not None
    ]
    marker = args.markers[index % len(args.markers)]
    ax.plot(
        x,
        y,
        label=label,
        # linestyle=(0, (1 + index // s, 1 + index % s)),
        linestyle=args.default_linestyles[index % len(args.default_linestyles)],
        markersize=4,
        marker=marker,
        color=c,
        fillstyle="none",
        alpha=0.85,
        linewidth=0.875,
    )
    if args.annotate:
        (ks,) = np.nonzero(y == np.max(y)) if y else (np.empty((0,), dtype=np.int64),)
        for k in ks:
            ax.annotate(
                "{:.4f} (tms: {}, c: {})".format(
                    y[k],
                    test_max_size_table_count_list[k][0],
                    test_max_size_table_count_list[k][1],
                ),
                xy=(x[k], y[k]),
                xytext=(
                    x[k],
                    y[k]
                    - 0.1
                    * (max(y, default=0) - min(y, default=0))
                    * (2 + (index * 3 + k * 7) % 5),
                ),
                arrowprops=dict(arrowstyle="->", color=c),
                color=c,
                alpha=0.85,
            )


def create_figure(epochs, args):
    fig = plt.figure(figsize=(8, 8 / args.aspect) if args.aspect else None)
    ax = fig.subplots()  # add_subplot(111, aspect="equal")
    space_for_legend = 0
    ax.set_xlim(args.min_epochs - 0.25, epochs + 0.25 + space_for_legend)
    ax.set_xticks(np.arange(epochs, args.min_epochs, -1), minor=True)
    # ax.set_xticks(np.arange((epochs + 4) // 5 * 5, args.min_epochs - 1, -5))
    ax.set_xticks(np.arange((args.min_epochs + 4) // 5 * 5, epochs + 1, 5))

    ax.tick_params(top=True, right=True, labelleft=False, labelright=True)
    ax.tick_params(
        top=True, right=True, labelleft=False, labelright=True, which="minor"
    )
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    if not args.vertical_axis_labels:
        ax.get_yaxis().set_ticks([])
    return fig, ax


def draw_plots(args):
    d = collections.defaultdict(list)
    for px_type, (log_pattern, logs) in {
        "msft": (args.msft_log_pattern, args.msft_logs),
        "std": (args.standard_log_pattern, args.standard_logs),
        "stdr": (args.standard_random_log_pattern, args.standard_random_logs),
        "fix_all": (args.fix_all_log_pattern, args.fix_all_logs),
    }.items():

        for file_path in logs:
            # print(file_path)
            if not file_path.exists():
                continue
            match = log_pattern.match(str(file_path))
            if not match:
                continue
            test_max_size = int(match.group("test_max_size"))
            epoch = int(match.group("epoch"))
            with open(file_path, mode="rt") as f:
                grits_perf = read_grits_performance(f)
            if grits_perf:
                update_list_test_max_size(
                    epoch - 1,
                    test_max_size,
                    grits_perf,
                    d["{} {:<3}".format(match.group("test_split_name"), px_type)],
                )
    for px_type, (log_pattern, logs) in {
        "pxc": (args.pxc_log_pattern, args.pxc_logs),
        "pxct": (args.pxct_log_pattern, args.pxct_logs),
        "pxctr": (args.pxct_random_log_pattern, args.pxct_random_logs),
    }.items():
        for file_path in logs:
            # print(file_path)
            if not file_path.exists():
                continue
            match = log_pattern.match(str(file_path))
            if not match:
                continue
            px = match.group("px")
            test_max_size = int(match.group("test_max_size"))
            epoch = int(match.group("epoch"))
            with open(file_path, mode="rt") as f:
                grits_perf = read_grits_performance(f)
            if grits_perf:
                update_list_test_max_size(
                    epoch - 1,
                    test_max_size,
                    grits_perf,
                    d[
                        "{} {:<3}-{}".format(
                            match.group("test_split_name"), px_type, px
                        )
                    ],
                )
    for px_type, (log_pattern, logs) in {
        "fix": (args.fix_log_pattern, args.fix_logs),
        "pxst": (args.pxst_log_pattern, args.pxst_logs),
    }.items():
        for file_path in logs:
            # print(file_path)
            if not file_path.exists():
                continue
            match = log_pattern.match(str(file_path))
            eps = match.group("eps")
            px = match.group("px")
            test_max_size = int(match.group("test_max_size"))
            epoch = int(match.group("epoch"))
            with open(file_path, mode="rt") as f:
                grits_perf = read_grits_performance(f)
            if grits_perf:
                update_list_test_max_size(
                    epoch - 1,
                    test_max_size,
                    grits_perf,
                    d[
                        "{} {:<3}-eps-{}-px-{}".format(
                            match.group("test_split_name"), px_type, eps, px
                        )
                    ],
                )
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    epochs = min(args.max_epochs, max((len(value) for value in d.values()), default=0))
    if args.text_usetex:
        plt.rcParams["text.usetex"] = True
    if args.split_by_table_type:
        for table_type in "simple", "complex", "all":
            for key in ["accuracy_con", "grits_top", "grits_con", "grits_loc"]:
                image_path = (
                    pathlib.Path(
                        args.output_dir,
                        "{}_{}_px_coco_metrics.svg".format(table_type, key),
                    )
                    if args.output_dir
                    else None
                )
                print(image_path)
                fig, ax = create_figure(epochs, args)
                for index, ((label, v), c) in enumerate(
                    zip(
                        # sorted(d.items()),
                        d.items(),
                        sns.color_palette(cc.glasbey, n_colors=len(d)),
                    )
                ):
                    draw_line(table_type, label, v, c, epochs, index, key, args, ax)
                ax.set(
                    xlabel="Epoch", ylabel=grits_name(key, text_usetex=args.text_usetex)
                )
                ax.legend(loc="lower right", fontsize="small", ncol=1)
                if args.output_dir:
                    fig.savefig(
                        image_path,
                        bbox_inches="tight",
                    )
                plt.close()
    else:
        for key in ["accuracy_con", "grits_top", "grits_con", "grits_loc"]:
            image_path = (
                pathlib.Path(args.output_dir, "{}_px_coco_metrics.svg".format(key))
                if args.output_dir
                else None
            )
            print(image_path)
            fig, ax = create_figure(epochs, args)
            # "TATR v1.1 with bug fixes" if label == "val std" else "Constrained box relaxation" if label == "val pxct-inf" else label,
            table_types = ("simple", "complex")
            for index, ((table_type, (label, v)), c) in enumerate(
                zip(
                    itertools.product(table_types, sorted(d.items())),
                    sns.color_palette(cc.glasbey, n_colors=len(d) * 3),
                )
            ):
                draw_line(
                    table_type,
                    "{}: {}".format(
                        table_type.title(),
                        (
                            "TATR v1.1 with bug fixes"
                            if label == "val std"
                            else (
                                "Constrained box relaxation"
                                if label == "val pxct-inf"
                                else label
                            )
                        ),
                    ),
                    v,
                    c,
                    epochs,
                    index,
                    key,
                    args,
                    ax,
                )
            ax.set(xlabel="Epoch", ylabel=grits_name(key, text_usetex=args.text_usetex))
            ax.legend(loc="lower right", fontsize="small", ncol=1)
            if args.output_dir:
                fig.savefig(
                    image_path,
                    bbox_inches="tight",
                )
            plt.close()


if __name__ == "__main__":
    parser = get_args_parser()
    draw_plots(parser.parse_args())
