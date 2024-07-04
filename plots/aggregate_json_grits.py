import argparse
import pathlib
import typing
import glob
import grits_metrics
import json
import os


def get_args_parser():
    parser = argparse.ArgumentParser("aggregate_json_grits", add_help=False)
    parser.add_argument(
        "--json_metrics_globs",
        type=str,
        nargs="*",
        help="Glob of files to be aggregated",
    )
    parser.add_argument("--intersect_with_filelist", type=pathlib.PurePath)
    parser.add_argument(
        "--output_path",
        type=pathlib.PurePath,
        help="Where to output aggregate simple/complex/all GriTS metrics.",
    )
    parser.add_argument(
        "--intersect_output_path",
        type=pathlib.PurePath,
        help="Where to output aggregate simple/complex/all GriTS metrics.",
    )
    parser.add_argument(
        "--check_unique", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


class RunningGrits(object):
    def __init__(self, table_count, grits_sum):
        self.table_count = table_count
        self.grits_sum = grits_sum

    def add_inplace(self, grits: grits_metrics.Grits):
        self.table_count += 1
        self.grits_sum = self.grits_sum.add(grits)

    def to_subset_grits(self):
        return grits_metrics.SubsetGrits(
            table_count=self.table_count, grits=self.grits_sum.divide(self.table_count)
        )

    def combine_with(self, other):
        return RunningGrits(
            table_count=self.table_count + other.table_count,
            grits_sum=self.grits_sum.add(other.grits_sum),
        )


class RunningGritsPerf(typing.NamedTuple):
    simple: RunningGrits
    complex: RunningGrits

    def to_grits_perf(self):
        return grits_metrics.GritsPerf(
            simple=self.simple.to_subset_grits(),
            complex=self.complex.to_subset_grits(),
            all=self.simple.combine_with(self.complex).to_subset_grits(),
        )


def aggregate(args):
    running_grits_perf = RunningGritsPerf(
        simple=RunningGrits(table_count=0, grits_sum=grits_metrics.Grits(0, 0, 0, 0)),
        complex=RunningGrits(table_count=0, grits_sum=grits_metrics.Grits(0, 0, 0, 0)),
    )
    if args.intersect_with_filelist:
        intersect_running_grits_perf = RunningGritsPerf(
            simple=RunningGrits(
                table_count=0, grits_sum=grits_metrics.Grits(0, 0, 0, 0)
            ),
            complex=RunningGrits(
                table_count=0, grits_sum=grits_metrics.Grits(0, 0, 0, 0)
            ),
        )
    if args.check_unique:
        ids = set()
    if args.intersect_with_filelist:
        with open(args.intersect_with_filelist, "rt") as f:
            allowed_ids = frozenset(pathlib.PurePath(line).stem for line in f)
    for json_metrics_glob in args.json_metrics_globs:
        for json_metrics_file in glob.iglob(json_metrics_glob):
            with open(json_metrics_file, "rt") as f:
                tsr_metrics_list = json.load(f)
            if args.check_unique:
                prev_size = len(ids)
                batch_ids = {d["id"] for d in tsr_metrics_list}
                assert len(batch_ids) == len(tsr_metrics_list)
                assert not ids.intersection(batch_ids), (json_metrics_file, ids.intersection(batch_ids))
                ids.update(batch_ids)
                assert len(ids) == prev_size + len(batch_ids)
            for d in tsr_metrics_list:
                grits = grits_metrics.Grits(
                    accuracy_con=d["acc_con"],
                    grits_top=d["grits_top"],
                    grits_con=d["grits_con"],
                    grits_loc=d["grits_loc"],
                )
                (
                    running_grits_perf.complex
                    if d["num_spanning_cells"]
                    else running_grits_perf.simple
                ).add_inplace(grits)
                if not args.intersect_with_filelist or d["id"] not in allowed_ids:
                    continue
                (
                    intersect_running_grits_perf.complex
                    if d["num_spanning_cells"]
                    else intersect_running_grits_perf.simple
                ).add_inplace(grits)

    grits_perf = running_grits_perf.to_grits_perf()
    if grits_perf.all.table_count and args.output_path:
        os.makedirs(args.output_path.parent, exist_ok=True)
        print(args.output_path)
        with open(args.output_path, "wt") as f:
            f.write(grits_perf.to_log_format())
    if not args.intersect_with_filelist:
        return
    intersect_grits_perf = intersect_running_grits_perf.to_grits_perf()
    if intersect_grits_perf.all.table_count and args.intersect_output_path:
        os.makedirs(args.intersect_output_path.parent, exist_ok=True)
        print(args.intersect_output_path)
        with open(args.intersect_output_path, "wt") as f:
            f.write(intersect_grits_perf.to_log_format())


if __name__ == "__main__":
    parser = get_args_parser()
    aggregate(parser.parse_args())
