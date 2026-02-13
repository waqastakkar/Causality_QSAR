#!/usr/bin/env python
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib as mpl

NATURE5 = ["#E69F00", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]


@dataclass(frozen=True)
class PlotStyle:
    font_family: str = "Times New Roman"
    font_title: int = 16
    font_label: int = 14
    font_tick: int = 12
    font_legend: int = 12
    bold_text: bool = True
    palette: Sequence[str] = tuple(NATURE5)


def configure_matplotlib(style: PlotStyle, svg: bool = True) -> None:
    if svg:
        mpl.rcParams["savefig.format"] = "svg"
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["font.family"] = style.font_family
    mpl.rcParams["font.weight"] = "bold" if style.bold_text else "normal"
    mpl.rcParams["axes.labelweight"] = "bold" if style.bold_text else "normal"
    mpl.rcParams["axes.titleweight"] = "bold" if style.bold_text else "normal"
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=list(style.palette))


def style_axis(ax, style: PlotStyle, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None) -> None:
    w = "bold" if style.bold_text else "normal"
    if title:
        ax.set_title(title, fontsize=style.font_title, fontweight=w)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=style.font_label, fontweight=w)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=style.font_label, fontweight=w)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(style.font_tick)
        tick.set_fontweight(w)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(style.font_legend)
            text.set_fontweight(w)


def parse_palette(palette: str) -> list[str]:
    if palette == "nature5":
        return NATURE5
    return [c.strip() for c in palette.split(",") if c.strip()]
