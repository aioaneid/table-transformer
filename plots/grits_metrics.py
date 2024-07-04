import typing
import textwrap


class Grits(typing.NamedTuple):
    accuracy_con: float
    grits_top: float
    grits_con: float
    grits_loc: float

    def add(self, other):
        return Grits(
            accuracy_con=self.accuracy_con + other.accuracy_con,
            grits_top=self.grits_top + other.grits_top,
            grits_con=self.grits_con + other.grits_con,
            grits_loc=self.grits_loc + other.grits_loc,
        )

    def divide(self, divisor):
        return (
            Grits(
                accuracy_con=self.accuracy_con / divisor,
                grits_top=self.grits_top / divisor,
                grits_con=self.grits_con / divisor,
                grits_loc=self.grits_loc / divisor,
            )
            if divisor
            else Grits(1, 1, 1, 1)
        )

    def to_log_format(self):
        return textwrap.dedent(
            """\
            Accuracy_Con: {}
               GriTS_Top: {}
               GriTS_Con: {}
               GriTS_Loc: {}"""
        ).format(self.accuracy_con, self.grits_top, self.grits_con, self.grits_loc)


SEPARATOR = "-" * 50


class SubsetGrits(typing.NamedTuple):
    table_count: int
    grits: Grits

    def to_log_format(self, subset_name):
        return textwrap.dedent(
            """\
            Results on {} tables ({} total):
            {}
            {}
            """
        ).format(
            subset_name,
            self.table_count,
            textwrap.indent(self.grits.to_log_format(), " " * 6),
            SEPARATOR,
        )


class GritsPerf(typing.NamedTuple):
    simple: SubsetGrits
    complex: SubsetGrits
    all: SubsetGrits

    def to_log_format(self):
        return "{}{}{}".format(
            self.simple.to_log_format("simple"),
            self.complex.to_log_format("complex"),
            self.all.to_log_format("all"),
        )
