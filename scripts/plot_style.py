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
    palette: Sequence[str] = tuple(NATURE5)


def configure_matplotlib(style: PlotStyle) -> None:
    mpl.rcParams["savefig.format"] = "svg"
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["font.family"] = style.font_family
    mpl.rcParams["font.weight"] = "bold"
    mpl.rcParams["axes.labelweight"] = "bold"
    mpl.rcParams["axes.titleweight"] = "bold"


def style_axis(ax, style: PlotStyle, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None) -> None:
    if title:
        ax.set_title(title, fontsize=style.font_title, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=style.font_label, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=style.font_label, fontweight="bold")

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(style.font_tick)
        tick.set_fontweight("bold")

    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(style.font_legend)
            text.set_fontweight("bold")
        legend.set_title(legend.get_title().get_text(), prop={"size": style.font_legend, "weight": "bold"})
