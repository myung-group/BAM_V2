"""Tabular training logger (port of ``bam_torch.utils.logger.Logger`` from BAM-torch-V2).

One header, then one line per epoch, grouped into column blocks:

    MM/DD/YYYY HH:MM:SS   EPOCH  TIME(s) | TRAIN ________________________ | VALID _______________________________ | LR
                                         | LOSS       E_LOSS     F_LOSS    | LOSS       E_RMSE     F_RMSE    ...   |
    -----------------------------------------------------------------------------------------------------------------
    10/26/2026 14:01:12   1      17.3    | 0.000750   0.000024   0.000726  | 0.000333   20.9       26.6      ...   | 0.01

Usage::

    logger = Logger(
        groups={"step": ["epoch", "time"],
                "train": ["loss", "E_loss", "F_loss", "grad"],
                "valid": ["loss", "E_RMSE", "F_RMSE", "E_MAE", "F_MAE"],
                "lr": ["lr"]},
        fout=fout, precision="simple")
    logger.print_head()
    logger.print_row({"epoch": 1, "time": 17.3}, {"loss": .., ...}, {"loss": .., ...}, {"lr": 1e-2})
    logger.message("free text line")          # goes to the same file (and stdout)
"""

from __future__ import annotations

import datetime
import sys
from typing import Dict, Iterable, List, Optional, TextIO


def date() -> str:
    return datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S")


class Logger:
    def __init__(
        self,
        groups: Dict[str, List[str]],
        fout: Optional[TextIO] = None,
        precision: str = "simple",
        echo: bool = True,
        units: Optional[Dict[str, str]] = None,
    ):
        """
        groups    : ordered dict  group name -> list of column names.  The first
                    group is the "step" block (printed with the date), the last
                    group is printed without a trailing divider.
        fout      : file to write to (None = file output disabled).
        precision : "simple" (5 significant digits) or "precise" (10).
        echo      : also print to stdout.
        units     : optional column -> unit suffix shown in the header, e.g. {"F_RMSE": "meV/A"}.
        """
        self.groups = {k: list(v) for k, v in groups.items()}
        self.fout = fout
        self.echo = echo
        self.units = units or {}
        self.sig = 5 if precision == "simple" else 10
        self.width = 11 if precision == "simple" else 16
        self.date_width = 21
        self.divider = " | "
        self._widths = self._column_widths()
        self.separator = "-" * len(self._format_header_line())

    # ------------------------------------------------------------------ layout
    def _column_widths(self) -> Dict[str, List[int]]:
        widths = {}
        for gi, (g, cols) in enumerate(self.groups.items()):
            w = []
            for c in cols:
                label = self._label(c)
                if gi == 0:
                    w.append(max(len(label) + 2, 8))
                else:
                    w.append(max(self.width, len(label) + 2))
            widths[g] = w
        return widths

    def _label(self, col: str) -> str:
        u = self.units.get(col)
        return f"{col}[{u}]" if u else col

    def _format_header_line(self) -> str:
        line = f"{'MM/DD/YYYY HH:MM:SS':<{self.date_width}}"
        parts = []
        for gi, (g, cols) in enumerate(self.groups.items()):
            s = "".join(f"{self._label(c).upper():<{w}}" for c, w in zip(cols, self._widths[g]))
            parts.append(s)
        line += parts[0]
        for s in parts[1:]:
            line += self.divider + s
        return line

    def _format_group_line(self) -> str:
        line = " " * self.date_width + " " * sum(self._widths[list(self.groups)[0]])
        for g in list(self.groups)[1:]:
            block_w = sum(self._widths[g])
            name = g.upper()
            line += self.divider + name + "_" * max(block_w - len(name), 0)
        return line

    # ----------------------------------------------------------------- output
    def _emit(self, text: str):
        if self.fout is not None:
            print(text, file=self.fout)
            self.fout.flush()
        if self.echo:
            print(text)
            sys.stdout.flush()

    def print_head(self):
        self._emit(self._format_group_line())
        self._emit(self._format_header_line())
        self._emit(self.separator)

    def _fmt(self, v, w: int, is_step: bool) -> str:
        if v is None:
            return f"{'-':<{w}}"
        if is_step:
            if isinstance(v, float):
                return f"{v:<{w}.1f}"
            return f"{v!s:<{w}}"
        try:
            return f"{float(v):<{w}.{self.sig}g}"
        except (TypeError, ValueError):
            return f"{v!s:<{w}}"

    def print_row(self, *values: Dict[str, object]):
        """One dict per group, in the order of ``groups``. Missing keys print '-'."""
        if len(values) != len(self.groups):
            raise ValueError(f"expected {len(self.groups)} dicts, got {len(values)}")
        line = f"{date():<{self.date_width}}"
        parts = []
        for gi, ((g, cols), vals) in enumerate(zip(self.groups.items(), values)):
            s = "".join(self._fmt(vals.get(c), w, gi == 0) for c, w in zip(cols, self._widths[g]))
            parts.append(s)
        line += parts[0]
        for s in parts[1:]:
            line += self.divider + s
        self._emit(line)

    def message(self, *lines: str):
        for l in lines:
            self._emit(l)

    def rule(self):
        self._emit(self.separator)
