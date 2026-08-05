# -*- coding: utf-8 -*-
"""This file contains standardized functions intended to simplify the
creation of publication-quality plots in a visually appealing, unique,
and consistent style.

Created on June 22nd, 2022

@author: mccambria
"""

# region Imports and constants

import itertools
import re
import string
import sys
from enum import Enum, auto

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from colorutils import Color
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from strenum import StrEnum
from matplotlib import font_manager as _fm
from pathlib import Path as _Path

import utils.common as common
# import data_manager as dm  # removed to avoid circular import

# from utils import data_manager

alphabet = tuple(string.ascii_lowercase)


# matplotlib semantic locations for legends and text boxes
class Loc(StrEnum):
    BEST = "best"
    LOWER_LEFT = "lower left"
    UPPER_LEFT = "upper left"
    LOWER_RIGHT = "lower right"
    UPPER_RIGHT = "upper right"
    UPPER_CENTER = "upper center"
    LOWER_CENTER = "lower center"
    CENTER_LEFT = "center left"
    CENTER_RIGHT = "center right"


class Size(Enum):
    BIG = "BIG"
    NORMAL = "NORMAL"
    SMALL = "SMALL"
    XSMALL = "XSMALL"
    TINY = "TINY"


class MarkerSize(float, Enum):
    BIG = 8
    NORMAL = 7
    SMALL = 6
    XSMALL = 5
    TINY = 3


class LineWidth(float, Enum):
    HUGE = 2.5
    BIG = 2.0
    NORMAL = 1.5
    SMALL = 1.25
    XSMALL = 1.1
    TINY = 0.8


class MarkerEdgeWidth(float, Enum):
    BIG = 1.75
    NORMAL = 1.5
    SMALL = 1.25
    XSMALL = 1.1
    TINY = 0.8


class FontSize(float, Enum):
    NORMAL = 17
    SMALL = 14
    TINY = 11


class PlotType(Enum):
    POINTS = auto()
    LINE = auto()
    HIST = auto()


class Font(Enum):
    ROBOTO = auto()
    HELVETICA = auto()


# Histogram type, mostly following https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
class HistType(Enum):
    INTEGER = auto()  # Just plot the frequency of each integer
    STEP = auto()  # No space between bins, translucent fill
    BAR = auto()  # Space between bins, filled


# Default sizes
marker_size = MarkerSize.NORMAL
marker_size_inset = MarkerSize.SMALL
line_width = LineWidth.NORMAL
line_width_inset = LineWidth.NORMAL
marker_edge_width = MarkerEdgeWidth.NORMAL
marker_edge_width_inset = MarkerEdgeWidth.NORMAL
default_font_size = Size.NORMAL
default_data_size = Size.NORMAL
figsize = [6.5, 5.0]
double_figsize = [figsize[0] * 2, figsize[1]]

# Default styles
line_style = "solid"
marker_style = "o"

# endregion
# region Colors
"""The default color specification is hex, eg "#bcbd22'"""


class KplColors(StrEnum):
    BLUE = "#1f77b4"
    RED = "#d62728"
    GREEN = "#2ca02c"
    ORANGE = "#ff7f0e"
    PURPLE = "#9467bd"
    BROWN = "#8c564b"
    PINK = "#e377c2"
    GRAY = "#7f7f7f"
    YELLOW = "#bcbd22"
    CYAN = "#17becf"
    #
    # The following are good for background lines on plots to mark interesting points
    MEDIUM_GRAY = "#C0C0C0"
    # If marking an interesting point with a confidence interval, use dark_gray for the main line and light_gray for the interval
    DARK_GRAY = "#909090"
    LIGHT_GRAY = "#DCDCDC"
    BLACK = "#000000"
    WHITE = "#FFFFFF"


data_color_cycler = [
    KplColors.BLUE,
    KplColors.RED,
    KplColors.GREEN,
    KplColors.ORANGE,
    KplColors.PURPLE,
    KplColors.BROWN,
    KplColors.PINK,
    KplColors.GRAY,
    KplColors.YELLOW,
    KplColors.CYAN,
    mpl.colors.cnames["darkgoldenrod"],
    # mpl.colors.cnames["greenyellow"],
    # mpl.colors.cnames["darkseagreen"],
    mpl.colors.cnames["indianred"],
    mpl.colors.cnames["darkslateblue"],
    # mpl.colors.cnames["sienna"],
]
line_color_cycler = data_color_cycler.copy()
hist_color_cycler = data_color_cycler.copy()


def color_mpl_to_color_hex(color_mpl):
    """Convert a named color from a matplotlib color map into hex"""
    # Trim the alpha value and convert from 0:1 to 0:255
    color_rgb = [255 * val for val in color_mpl[0:3]]
    color_Color = Color(tuple(color_rgb))
    color_hex = color_Color.hex
    return color_hex


def lighten_color_hex(color_hex, saturation_factor=0.3, value_factor=1.2):
    """Algorithmically lighten the passed hex color"""

    color_Color = Color(hex=color_hex)
    color_hsv = color_Color.hsv
    lightened_hsv = (
        color_hsv[0],
        saturation_factor * color_hsv[1],
        value_factor * color_hsv[2],
    )
    # Threshold to make sure these are valid colors
    lightened_hsv = (
        lightened_hsv[0],
        zero_to_one_threshold(lightened_hsv[1]),
        zero_to_one_threshold(lightened_hsv[2]),
    )
    lightened_Color = Color(hsv=lightened_hsv)
    lightened_hex = lightened_Color.hex
    return lightened_hex


def alpha_color_hex(color_hex, alpha=0.3):
    """Algorithmically drop the alpha on the passed hex color (i.e. make it translucent)"""

    hex_alpha = hex(round(alpha * 255))
    if len(hex_alpha) == 3:
        hex_alpha = f"0{hex_alpha[-1]}"
    else:
        hex_alpha = hex_alpha[-2:]
    return f"{color_hex}{hex_alpha}"


def zero_to_one_threshold(val):
    """Clip the passed value such that it is between 0 and 1"""
    if val < 0:
        return 0
    elif val > 1:
        return 1
    else:
        return val


# endregion
# region Miscellaneous

kplotlib_initialized = False


def calc_mosaic_layout(num_panels, num_rows=None, num_cols=None):
    if num_rows is None and num_cols is None:
        num_rows = round(np.sqrt(num_panels))
        num_cols = int(np.ceil(num_panels / num_rows))
    elif num_cols is None:
        num_cols = int(np.ceil(num_panels / num_rows))
    elif num_rows is None:
        num_rows = int(np.ceil(num_panels / num_cols))
    num_axes = num_cols * num_rows

    vals = np.array(
        [[f"{row}{col}" for col in alphabet[:num_cols]] for row in alphabet[:num_rows]]
    )
    if num_panels != num_axes:
        vals[0, num_panels - num_axes :] = "."

    return vals
    # return vals.tolist()


def subplot_mosaic(num_panels, num_rows=None, figsize=[10, 6.0], **kwargs):
    layout = calc_mosaic_layout(num_panels, num_rows)
    fig, axes_pack = plt.subplot_mosaic(
        layout, figsize=figsize, sharex=True, sharey=True, **kwargs
    )
    return fig, axes_pack, layout


def set_shared_ax_xlabel(lower_left_ax, label, **kwargs):
    _set_shared_ax_axis_label(True, lower_left_ax, label, **kwargs)


def set_shared_ax_ylabel(lower_left_ax, label, **kwargs):
    _set_shared_ax_axis_label(False, lower_left_ax, label, **kwargs)


def _set_shared_ax_axis_label(x_or_y, lower_left_ax, label, **kwargs):
    """Works by making a dummy axis just for setting the label"""
    ax = lower_left_ax
    fig = ax.get_figure()
    try:
        label_ax = fig.label_ax
    except Exception:
        label_ax = fig.add_subplot(frameon=False)
        fig.label_ax = label_ax
        label_ax.tick_params(
            which="both",
            top=False,
            bottom=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )
    if x_or_y:
        ax.set_xlabel(" ")
        label_ax.set_xlabel(label, **kwargs)
    else:
        ax.set_ylabel(" ")
        label_ax.set_ylabel(label, **kwargs)


def _register_local_roboto():
    base = _Path(__file__).resolve().parent / "assets/fonts/roboto"
    if base.exists():
        for ttf in base.glob("*.ttf"):
            _fm.fontManager.addfont(str(ttf))
        _fm._rebuild()  # refresh in-memory font cache


def init_kplotlib(
    font_size=Size.NORMAL,
    data_size=Size.NORMAL,
    font=Font.ROBOTO,
    constrained_layout=True,
    latex=False,
    force=False,  # <-- added by SC
):
    """Runs the initialization for kplotlib, our default configuration
    of matplotlib. Plotting will be faster if latex is False - only set to True
    if you need full access to LaTeX
    """

    ### Misc setup

    global kplotlib_initialized
    # if kplotlib_initialized:
    #     return
    if kplotlib_initialized and not force:  # <-- use force
        return
    kplotlib_initialized = True

    # Reset to the default
    mpl.rcParams.update(mpl.rcParamsDefault)

    global active_axes, color_cyclers, default_font_size, default_data_size
    active_axes = []
    color_cyclers = []
    default_font_size = font_size
    default_data_size = data_size

    # Interactive mode so plots update as soon as the event loop runs
    plt.ion()

    ### Latex setup

    preamble = r""
    preamble += r"\newcommand\hmmax{0} \newcommand\bmmax{0}"
    preamble += r"\usepackage{physics} \usepackage{upgreek}"
    if font == Font.ROBOTO:
        preamble += r"\usepackage{roboto}"  # Google's free Helvetica
    elif font == Font.HELVETICA:
        preamble += r"\usepackage{helvet}"

    preamble += r"\usepackage[T1]{fontenc}"
    preamble += r"\usepackage{siunitx}"
    preamble += r"\sisetup{detect-all}"

    # Note: The global usetex setting should remain off. This prevents serif fonts
    # from proliferating (e.g. to axis tick labels). Instead just pass usetex=True
    # as a kwarg to any text-based command as necessary. If really necessary, flip
    # the flag below and use the \mathrm and \mathit macros to keep serifs in check
    if False:  # Global usetex?
        preamble += r"\usepackage[mathrmOrig, mathitOrig]{sfmath}"
        plt.rcParams["text.latex.preamble"] = preamble
        plt.rcParams["text.usetex"] = True
        plt.rc("text", usetex=True)
    else:
        plt.rcParams["text.latex.preamble"] = preamble

    ### Other rcparams
    # plt.rcParams["legend.handlelength"] = 0.5
    # plt.rcParams["font.family"] = "sans-serif"
    # if font == Font.ROBOTO:
    #     plt.rcParams["font.sans-serif"] = "Roboto"
    # if font == Font.HELVETICA:
    #     plt.rcParams["font.sans-serif"] = "Helvetica"
    plt.rcParams["font.family"] = "sans-serif"
    if font == Font.ROBOTO:
        _register_local_roboto()
        plt.rcParams["font.sans-serif"] = [
            "Roboto",
            "DejaVu Sans",
            "Arial",
            "Liberation Sans",
        ]
    elif font == Font.HELVETICA:
        plt.rcParams["font.sans-serif"] = [
            "Helvetica",
            "Arial",
            "DejaVu Sans",
            "Liberation Sans",
        ]
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]

    plt.rcParams["font.size"] = FontSize[default_font_size.value]
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.format"] = "png"  # "svg"
    plt.rcParams["figure.max_open_warning"] = 100
    plt.rcParams["image.cmap"] = "inferno"
    plt.rcParams["figure.constrained_layout.use"] = constrained_layout
    plt.rcParams["image.interpolation"] = "nearest"
    plt.rcParams["legend.fontsize"] = 0.9 * FontSize[default_font_size.value]
    plt.rcParams["legend.handlelength"] = 0.8
    plt.rcParams["legend.handletextpad"] = 0.4
    plt.rcParams["legend.columnspacing"] = 1.0
    plt.rcParams["legend.borderaxespad"] = 0.2
    plt.rcParams["legend.borderpad"] = 0.3
    plt.rcParams["axes.ymargin"] = 0.02
    plt.rcParams["axes.axisbelow"] = False


def get_default_color(ax, plot_type):
    """Get the default color according to the cycler of the passed plot type.

    plot_type : PlotType(enum)
    """

    global active_axes, color_cyclers
    if ax not in active_axes:
        active_axes.append(ax)
        color_cyclers.append(
            {
                PlotType.POINTS: data_color_cycler.copy(),
                PlotType.LINE: line_color_cycler.copy(),
                PlotType.HIST: hist_color_cycler.copy(),
            }
        )
    ax_ind = active_axes.index(ax)
    cycler = color_cyclers[ax_ind][plot_type]
    color = cycler.pop(0)
    return color


def anchored_text(ax, text, loc=Loc.UPPER_RIGHT, size=None, **kwargs):
    """Add text in default style to the passed ax. To update text call set_text on the
    returned object's txt property

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add the text box to
    text : str
    loc : str or Loc(enum)
        Relative location to anchor the text box to
    size : Size(enum), optional
        Font size, defaults to default_font_size set up in init_kplotlib

    Returns
    -------
    matplotlib.offsetbox.AnchoredText
    """

    global default_font_size
    if size is None:
        size = default_font_size

    font_size = FontSize[size.value]
    text_props = kwargs
    text_props["fontsize"] = font_size
    # text_props = dict(fontsize=font_size)
    text_box = AnchoredText(text, loc, prop=text_props)
    text_box.patch.set_boxstyle("round, pad=0, rounding_size=0.2")
    text_box.patch.set_facecolor("wheat")
    text_box.patch.set_alpha(0.5)
    ax.add_artist(text_box)
    return text_box


def legend(ax, usetex=False, *args, **kwargs):
    global_usetex = plt.rcParams["text.usetex"]
    plt.rcParams["text.usetex"] = usetex
    ax.legend(*args, **kwargs)
    plt.rcParams["text.usetex"] = global_usetex


def scale_bar(ax, length, label, loc):
    ylim = ax.get_ylim()
    size_vertical = 0.01 * (max(ylim) - min(ylim))
    bar = AnchoredSizeBar(
        ax.transData,
        length,
        label,
        loc,
        size_vertical=size_vertical,
        borderpad=0.2,
        pad=0.2,
        sep=4,
    )
    ax.add_artist(bar)
    return bar


def tex_escape(text):
    """Escape TeX characters in the passed text"""
    conv = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\^{}",
        "\\": r"\textbackslash{}",
        "<": r"\textless{}",
        ">": r"\textgreater{}",
    }
    regex = re.compile(
        "|".join(
            re.escape(str(key))
            for key in sorted(conv.keys(), key=lambda item: -len(item))
        )
    )
    return regex.sub(lambda match: conv[match.group()], text)


def show(block=False):
    """
    Show the current figures. Also processes any pending updates to the figures
    """
    fig = plt.gcf()
    fig.canvas.flush_events()
    plt.show(block=block)


# endregion
# region Plotting


def plot_points(ax, x, y, yerr=None, size=None, **kwargs):
    """Same as matplotlib's errorbar, but with our defaults. Use for plotting
    data points

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    x : 1D array
        x values to plot
    y : 1D array
        y values to plot
    size : Size(enum), optional
        Data point size, by default data_size passed to init_kplotlib
    kwargs
        Passed on to matplotlib's errorbar function
    """

    global default_data_size
    if size is None:
        size = default_data_size

    # Color handling
    if "color" in kwargs:
        color = kwargs["color"]
    else:
        color = get_default_color(ax, PlotType.POINTS)
    if "markerfacecolor" in kwargs:
        face_color = kwargs["markerfacecolor"]
    else:
        # if size is Size.TINY:
        #     face_color = color
        # else:
        #     face_color = lighten_color_hex(color)
        face_color = lighten_color_hex(color)

    # Defaults
    params = {
        "yerr": yerr,
        "linestyle": "none",
        "marker": marker_style,
        "markersize": MarkerSize[size.value],
        "markeredgewidth": MarkerEdgeWidth[size.value],
    }

    # Combine passed args and defaults
    params = {**params, **kwargs}
    params["color"] = color
    params["markerfacecolor"] = face_color

    ax.errorbar(x, y, **params)


def plot_sequence(ax, edges, values, size=None, **kwargs):
    global default_data_size
    if size is None:
        size = default_data_size

    # Color handling
    if "color" in kwargs:
        edge_color = kwargs["color"]
    else:
        edge_color = get_default_color(ax, PlotType.HIST)
    if "facecolor" in kwargs:
        face_color = kwargs["facecolor"]
    else:
        face_color = lighten_color_hex(edge_color)

    # Defaults
    params = {
        "fill": True,
        "linewidth": 1.2 * MarkerEdgeWidth[size.value],
        "linestyle": "-",
    }

    # Combine passed args and defaults
    params = {**params, **kwargs}
    params["edgecolor"] = edge_color
    params["facecolor"] = face_color

    ax.stairs(values, edges, **params)


def plot_bars(ax, x, y, **kwargs):
    """Same as matplotlib's bar, but with our defaults. Use for plotting
    data points

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    x : 1D array
        x values to plot
    y : 1D array
        y values to plot
    kwargs
        Passed on to matplotlib's errorbar function
    """

    # Color handling
    if "color" in kwargs:
        color = kwargs["color"]
    else:
        color = get_default_color(ax, PlotType.POINTS)
    if "markerfacecolor" in kwargs:
        face_color = kwargs["markerfacecolor"]
    else:
        face_color = lighten_color_hex(color)

    # Combine passed args and defaults
    params = kwargs
    params["ecolor"] = color
    params["edgecolor"] = color
    params["facecolor"] = face_color
    linewidth = 1.5
    params["linewidth"] = linewidth
    params["error_kw"] = {"elinewidth": linewidth}

    ax.bar(x, y, **params)


def plot_line(ax, x, y, size=None, **kwargs):
    """Same as matplotlib's plot, but with our defaults. Use for plotting
    continuous lines

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    x : 1D array
        x values to plot
    y : 1D array
        y values to plot
    size : Size(enum), optional
        Line width, by default data_size passed to init_kplotlib
    kwargs
        Passed on to matplotlib's plot function
    """

    global default_data_size
    if size is None:
        size = default_data_size

    # Color handling
    if "color" in kwargs:
        color = kwargs["color"]
    else:
        color = get_default_color(ax, PlotType.LINE)

    # Defaults
    params = {
        "linestyle": line_style,
        "linewidth": LineWidth[size.value],
    }

    # Combine passed args and defaults
    params = {**params, **kwargs}
    params["color"] = color

    ax.plot(x, y, **params)


def plot_line_update(ax, line_ind=0, x=None, y=None, relim_x=True, relim_y=True):
    """Updates a plot created by plot_line

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to update
    line_ind : int, optional
        Index of the line in the plot to update, by default 0
    x : 1D array, optional
        New x values to write, by default None
    y : 1D array, optional
        New y values to write, by default None
    relim_x : bool, optional
        Update the x limits of the plot? By default True
    relim_y : bool, optional
        Update the y limits of the plot? By default True
    """

    lines = ax.get_lines()
    line = lines[line_ind]

    if x is not None:
        line.set_xdata(x)
    if y is not None:
        line.set_ydata(y)

    if relim_x or relim_y:
        ax.relim()
        ax.autoscale_view(scalex=relim_x, scaley=relim_y)

    # Call show to update the actual display
    show()


def imshow(
    ax,
    img_array,
    title=None,
    x_label=None,
    y_label=None,
    cbar_label=None,
    no_cbar=False,
    nan_color=None,
    interpolation="none",
    **kwargs,
):
    """Same as matplotlib's imshow, but with our defaults

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes for the image
    img_array : 2D array
        Values to plot in the image
    title : str, optional
        Image title, by default None
    x_label : str, optional
        x axis label, by default None
    y_label : str, optional
        y axis label, by default None
    cbar_label : str, optional
        Color bar label, by default None
    kwargs
        Passed on to matplotlib's plot function

    Returns
    -------
    matplotlib.image.AxesImage
    """

    fig = ax.get_figure()

    img = ax.imshow(img_array, interpolation=interpolation, **kwargs)

    # Colorbar
    if not no_cbar:
        clb = fig.colorbar(img, ax=ax)
        if cbar_label is not None:
            clb.set_label(cbar_label)
    if nan_color is not None:
        cmap = img.cmap
        cmap.set_bad(color=nan_color)

    # Labels
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

    # Click handler
    fig.canvas.mpl_connect("button_press_event", on_click_image)

    return img


def imshow_update(ax, img_array, cmin=None, cmax=None):
    """Update the (first) image in the passed ax

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes for the image
    img_array : 2D array
        Values to plot in the image
    cmin : numeric, optional
        Minimum color bar value, by default None
    cmax : numeric, optional
        Maximum color bar value, by default None
    """
    images = ax.get_images()
    img = images[0]
    img.set_data(img_array)
    if (cmin != None) & (cmax != None):
        img.set_clim(cmin, cmax)
    else:
        img.autoscale()
    show()


def on_click_image(event):
    """Click handler for images from imshow. Prints the click coordinates to
    the console. Event is passed automatically
    """
    x_coord = round(event.xdata, 3)
    y_coord = round(event.ydata, 3)
    try:
        print(f"{x_coord}, {y_coord}")
    except TypeError:
        # Ignore the exc that's raised if the user clicks out of bounds
        pass


def histogram(ax, data, hist_type=HistType.INTEGER, nbins=None, bin_size=1, **kwargs):
    """Similar to matplotlib's hist, but with our defaults

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    data : array
        Data to histogram - will be flattened
    hist_type : HistType(enum), optional
        Histogram type, by default HistType.INTEGER
    nbins : int, optional
        Number of bins, by default calculated automatically
    kwargs
        Passed on to matplotlib's plot function

    Returns
    -------
    1D array
        Histogram occurrences (same as numpy histogram)
    1D array
        Histogram bin edges
    """

    # Color handling
    if "color" in kwargs:
        color = kwargs["color"]
    else:
        color = get_default_color(ax, PlotType.HIST)

    # Copy kwargs and set the color
    params = {**kwargs}
    params["color"] = color

    # For an integer histogram make a step histogram with the frequency of each integer
    # nbins does nothing here
    if hist_type == HistType.INTEGER:
        data = [round(el) for el in data]
        max_data = max(data)
        if bin_size == 1:
            rng = (-0.5, max_data + 0.5)
            nbins = max_data + 1
        else:
            nbins = int(np.ceil(max_data / bin_size))
            rng = (0, bin_size * nbins)
    else:
        rng = None
    if hist_type == HistType.INTEGER or hist_type == HistType.STEP:
        if "facecolor" in kwargs:
            facecolor = kwargs["facecolor"]
            del kwargs["facecolor"]
        else:
            facecolor = alpha_color_hex(color)
        occur, bin_edges, _ = ax.hist(
            data,
            histtype="step",
            bins=nbins,
            facecolor=facecolor,
            fill=True,
            range=rng,
            **kwargs,
        )
    elif hist_type == HistType.BAR:
        occur, bin_edges, _ = ax.hist(data, histtype="bar", bins=nbins, **kwargs)

    return occur, bin_edges


# def draw_circle(ax, coords, radius=1, color=KplColors.BLUE, outline=False, label=None):
#     """Draw a circle on the passed axes

#     Parameters
#     ----------
#     ax : matplotlib axes
#         Axes to draw the circle on
#     coords : 2-tuple
#         Center coordinates of the circle
#     radius : numeric
#         Radius of the circle
#     """
#     if outline:
#         circle = plt.Circle(coords, 1.5 * radius, color=KplColors.WHITE)
#         ax.add_artist(circle)
#     circle = plt.Circle(coords, radius, color=color, label=label)
#     ax.add_artist(circle)


def draw_circle(
    ax,
    coords,
    radius=1,
    color=KplColors.BLUE,
    label=None,
    linewidth=None,
    linestyle="solid",
):
    """Draw a circle on the passed axes

    Parameters
    ----------
    ax : matplotlib axes
        Axes to draw the circle on
    coords : 2-tuple
        Center coordinates of the circle
    radius : numeric
        Radius of the circle
    """
    M = ax.transData.get_matrix()
    xscale = M[0, 0]
    if linewidth is None:
        linewidth = 0.08 * xscale * radius
    ax.scatter(  # MCC
        *coords,
        s=(xscale * radius) ** 2,
        facecolors="none",
        edgecolors="black",
        label=label,
        linewidths=1.8 * linewidth,
        linestyle="solid",
        zorder=9,
    )
    return ax.scatter(
        *coords,
        s=(xscale * radius) ** 2,
        facecolors="none",
        edgecolors=color,
        label=label,
        linewidths=linewidth,
        linestyle=linestyle,
        zorder=10,
    )
    # circle = plt.Circle(
    #     coords, radius, fill=False, color=color, label=label, linewidth=linewidth
    # )
    # ax.add_patch(circle)


# endregion

if __name__ == "__main__":
    init_kplotlib()
    # fmt: off
    snr_list = [0.207, 0.206, 0.211, 0.183, 0.08, 0.224, 0.095, 0.078, 0.136, 0.038, 0.034, 0.026, 0.039, 0.165, 0.13, 0.18, 0.153, 0.074, 0.08, 0.028, 0.053, 0.142, 0.188, 0.077, 0.121, 0.137, 0.085, 0.067, 0.157, 0.135, 0.036, 0.075, 0.135, 0.168, 0.045, 0.067, 0.158, 0.12, 0.074, 0.167, 0.073, 0.046, 0.149, 0.054, 0.135, 0.064, 0.119, 0.193, 0.104, 0.091, 0.04, 0.127, 0.125, 0.105, 0.054, 0.069, 0.139, 0.151, 0.119, 0.068, 0.134, 0.054, 0.11, 0.096, 0.105, 0.133, 0.149, 0.057, 0.102, 0.083, 0.097, 0.175, 0.096, 0.058, 0.161, 0.158, 0.048, 0.1, 0.093, 0.132, 0.131, 0.055, 0.028, 0.083, 0.05, 0.061, 0.06, 0.082, 0.114, 0.065, 0.144, 0.142, 0.116, 0.095, 0.143, 0.121, 0.116, 0.102, 0.032, 0.061, 0.113, 0.087, 0.061, 0.119, 0.027, 0.119, 0.131, 0.144, 0.122, 0.087, 0.087, 0.067, 0.089, 0.068, 0.089, 0.043, 0.131, 0.05, 0.075, 0.039, 0.09, 0.085, 0.099, 0.123, 0.133, 0.097, 0.083, 0.04, 0.097, 0.032, 0.043, 0.148, 0.092, 0.037, 0.118, 0.051, 0.078, 0.053, 0.081, 0.056, 0.112, 0.119, 0.05, 0.044, 0.131, 0.137, 0.133, 0.074, 0.049, 0.06, 0.043, 0.063, 0.106, 0.165, 0.16, 0.05, 0.132, 0.088, 0.081, 0.062]
    scc_duration_list = [304, 304, 304, 156, 304, 148, 244, 100, 304, 60, 304, 76, 88, 304, 112, 304, 144, 304, 304, 48, 76, 140, 144, 88, 304, 304, 304, 112, 304, 172, 304, 96, 72, 168, 128, 48, 304, 112, 124, 304, 48, 304, 304, 48, 304, 304, 168, 144, 304, 304, 60, 304, 108, 304, 48, 304, 164, 160, 304, 268, 240, 196, 304, 112, 304, 48, 264, 304, 152, 304, 184, 148, 304, 52, 160, 112, 104, 304, 88, 116, 56, 304, 68, 304, 304, 112, 52, 304, 304, 96, 304, 120, 304, 140, 304, 304, 156, 48, 304, 64, 304, 304, 132, 124, 304, 148, 304, 148, 80, 136, 124, 148, 108, 132, 132, 68, 124, 132, 304, 92, 80, 64, 304, 152, 136, 304, 48, 96, 304, 48, 64, 304, 64, 304, 216, 304, 304, 144, 176, 140, 304, 136, 104, 304, 56, 136, 76, 112, 304, 120, 164, 304, 88, 104, 128, 152, 132, 112, 100, 304]
    scc_amp_list = [1.107, 1.071, 1.214, 1.179, 1.036, 1.179, 0.75, 1.214, 0.857, 0.857, 1.036, 0.821, 0.857, 1.25, 1.036, 1.179, 1.25, 0.786, 1.179, 1.036, 1.25, 1.0, 1.071, 1.25, 1.25, 0.857, 1.036, 1.071, 1.036, 1.143, 1.036, 0.75, 1.214, 0.964, 1.036, 0.75, 0.786, 0.964, 1.107, 0.857, 1.179, 0.857, 1.214, 1.143, 1.071, 1.25, 1.143, 0.857, 1.214, 1.143, 0.786, 0.929, 0.75, 1.071, 0.857, 0.75, 1.036, 1.071, 0.786, 1.107, 1.071, 1.214, 0.964, 0.929, 1.107, 1.143, 1.214, 1.071, 1.036, 1.214, 0.893, 1.071, 0.75, 0.786, 1.25, 1.107, 0.929, 0.786, 0.929, 1.25, 1.107, 1.036, 1.0, 0.893, 1.0, 0.964, 1.107, 1.143, 1.25, 1.214, 0.821, 0.929, 1.107, 1.107, 1.25, 1.214, 0.75, 1.214, 1.0, 1.25, 0.964, 0.857, 0.929, 1.25, 0.893, 1.0, 0.75, 1.179, 1.25, 1.214, 1.036, 0.821, 1.214, 1.0, 1.179, 1.214, 1.107, 1.25, 0.929, 1.036, 1.143, 0.821, 0.893, 1.179, 1.143, 0.893, 1.25, 1.071, 0.786, 1.25, 1.107, 1.179, 0.929, 1.0, 1.25, 0.964, 1.036, 1.036, 1.25, 1.179, 1.143, 1.143, 1.143, 1.179, 1.179, 1.143, 1.214, 1.107, 0.893, 1.25, 1.143, 0.964, 1.25, 1.036, 0.857, 1.107, 1.179, 1.0, 1.214, 0.786]
    # fmt: on

    raw_data = dm.get_raw_data(file_id=1723161184641)
    orientation_a_inds = raw_data["orientation_indices"]["0.041"]["nv_indices"]
    orientation_b_inds = raw_data["orientation_indices"]["0.147"]["nv_indices"]
    orientation_ab_inds = orientation_a_inds + orientation_b_inds

    num_nvs = len(snr_list)
    good_list = [
        scc_duration_list[ind]
        for ind in range(num_nvs)
        if snr_list[ind] > 0.07 and ind in orientation_ab_inds
    ]
    fig, ax = plt.subplots()
    histogram(ax, good_list, HistType.STEP)
    ax.set_xlabel("Duration (ns)")
    ax.set_ylabel("Number of occurrences")
    ax.set_title("All NVs")
    ax.set_title("All NVs")
    ax.set_title("Orientation B")
    show(block=True)
