#!/usr/bin/env python3
# encoding: utf-8

import matplotlib.ticker as tck
from matplotlib import gridspec as gs
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import os
import os.path as osp

from gridspeccer import core
from gridspeccer.core import log
from gridspeccer import aux


def get_gridspec():
    """
        Return dict: plot -> gridspec
    """

    gs_main = gs.GridSpec(1, 3,
                          left=0.05, right=0.97, top=0.90, bottom=0.12,
                          width_ratios=[1.2, 1.2, 1.0],
                          wspace=0.5,
                          )

    return {
        # ### schematics
        "arch": gs_main[0, 0],

        "umem": gs_main[0, 1],

        "raster": gs_main[0, 2],

    }


def adjust_axes(axes):
    """
        Settings for all plots.
    """
    for ax in axes.values():
        core.hide_axis(ax)

    for k in [
        "arch",
    ]:
        axes[k].set_frame_on(False)


def plot_labels(axes):
    core.plot_labels(axes,
                     labels_to_plot=[
                         "arch",
                         "umem",
                         "raster",
                     ],
                     label_ypos={
                         "arch": 1.02,
                         "umem": 1.02,
                         "raster": 1.02,
                     },
                     label_xpos={
                         "arch": -0.12,
                         "umem": -0.09,
                         "raster": -0.08,
                     },
                     label_size=12,
                     )


def get_fig_kwargs():
    width = 7.12
    alpha = 0.3
    return {"figsize": (width, alpha * width)}


###############################
# Plot functions for subplots #
###############################
#
# naming scheme: plot_<key>(ax)
#
# ax is the Axes to plot into
#
plotEvery = 1
xlim = (0, 100)

name_of_vleak = "$E_\ell$"
name_of_vth = "$\\vartheta$"
exampleClass = 1


def plot_arch(ax):
    # done with tex
    return


def plot_raster(ax):
    # make the axis
    core.show_axis(ax)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("time [a.u.]")
    ax.xaxis.set_label_coords(.5, -0.05)
    ax.set_ylabel("neuron id")
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # #################### draw arrow instead of y axis to have arrow there
    # get width and height of axes object to compute
    # matching arrowhead length and width
    # fig = plt.gcf()
    # dps = fig.dpi_scale_trans.inverted()

    # bbox = ax.get_window_extent()  # .transformed(dps)
    # width, height = bbox.width, bbox.height

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # manual arrowhead width and length
    hw = 1. / 20. * (ymax - ymin)
    hl = 1. / 20. * (xmax - xmin)
    lw = 0.5  # axis line width
    ohg = 0.3  # arrow overhang

    # compute matching arrowhead length and width
    # yhw = hw / (ymax - ymin) * (xmax - xmin) * height / width
    # yhl = hl / (xmax - xmin) * (ymax - ymin) * width / height

    ax.arrow(xmin, ymin, xmax - xmin, 0, fc='k', ec='k', lw=lw,
             head_width=hw, head_length=hl, overhang=ohg,
             length_includes_head=True, clip_on=False)
    return


def plot_umem(ax):
    core.show_axis(ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel(r"$u_\mathrm{mem}$ [a. u.]")
    ax.yaxis.set_label_coords(-0.03, 0.5)
    ax.set_xlabel("time [a. u.]")
    ax.xaxis.set_label_coords(.5, -0.05)

    xmax = 42.
    ymin = -0.08
    ymax = 0.25
    ax.set_xlim(0, xmax)
    ax.set_ylim(ymin, ymax)
    xvals = np.linspace(0., xmax, 200)
    c_m = 0.2
    t_s = 2.
    t_m = 4.
    w = 0.1
    v_th = 0.172

    t1 = 25
    t2 = 30
    t3 = 35
    t_single = 5
    t_delay = 10
    offsets = np.array([0. for x in xvals])
    u_mem = psp(xvals.copy(), t_single, t_s, t_m, w, offsets, v_th=v_th)
    ax.plot(xvals[:120], u_mem[:120], color='grey', alpha=0.8, ls='--')
    offsets = np.array([0. for x in xvals])
    u_mem = psp(xvals.copy(), t_single + t_delay, t_s, t_m, w, offsets, v_th=v_th)
    u_mem = psp(xvals.copy(), t1, t_s, t_m, w, u_mem, v_th=v_th)
    u_mem = psp(xvals.copy(), t2, t_s, t_m, w, u_mem, v_th=v_th)
    u_mem = psp(xvals.copy(), t3, t_s, t_m, w, u_mem, v_th=v_th)
    ax.plot(xvals, u_mem, color='k', alpha=0.8)

    t = 36.7
    ax.axvline(t, ymin=0.85, ymax=0.92, color='k', alpha=0.8)

    ax.arrow(t_single + 2.0, 0.01, t_delay - 3.5, 0, length_includes_head=False, color='C3', head_width=0.009, head_length=0.80, linewidth=1.5, zorder=5)
    ax.arrow(t_single + t_delay - 2.0, 0.01, -t_delay + 3.5, 0, length_includes_head=False, color='C3', head_width=0.009, head_length=0.80, linewidth=1.5, zorder=5)
    ax.text(9.0, 0.02, r"$d$", color='C3')
    ax.axvline(t_single, ymin=0.05, ymax=0.12, color='k', alpha=0.8)
    ax.axvline(t1, ymin=0.07, ymax=0.14, color='k', alpha=0.8)
    ax.axvline(t2, ymin=0.07, ymax=0.14, color='k', alpha=0.8)
    ax.axvline(t3, ymin=0.07, ymax=0.14, color='k', alpha=0.8)
    t_arrow = 17.8
    dy = 0.08
    ax.arrow(t_arrow, 0.007, 0, dy, length_includes_head=False, color='C3', head_width=0.9, head_length=0.0080, linewidth=1.5, zorder=5)
    ax.arrow(t_arrow, dy, 0, -dy + 0.010, length_includes_head=False, color='C3', head_width=0.9, head_length=0.0080, linewidth=1.5, zorder=5)
    ax.text(18.5, 0.04, r"$w$", color='C3')


def membrane_schematic(ax, should_spike, xAnnotated=False, yAnnotated=False):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    ylim = (-0.2, 1.2)

    xvals = np.linspace(0., 100., 200)
    c_m = 0.2
    t_s = 10.
    t_m = t_s
    w = 0.032
    t_i1 = 20.
    V_th = 1.

    def theta(x):
        return x > 0

    if should_spike:
        t_i2 = 30.
        t_spike = 35.

        def V(t, before_spike):
            return w / c_m * (
                theta(t - t_i1) * np.exp(-(t - t_i1) / t_m) * (t - t_i1) +
                theta(t - t_i2) * np.exp(-(t - t_i2) / t_m) * (t - t_i2)
            ) * ((t < t_spike) * before_spike + (t > t_spike) * (1 - before_spike))

        ax.plot(xvals, V(xvals, True), color='black')
        ax.plot(xvals[xvals > t_spike], V(xvals[xvals > t_spike], False), color='black', linestyle="dotted")
    else:
        t_i2 = 35.

        def V(t):
            return w / c_m * (
                theta(t - t_i1) * np.exp(-(t - t_i1) / t_m) * (t - t_i1) +
                theta(t - t_i2) * np.exp(-(t - t_i2) / t_m) * (t - t_i2)
            )

        ax.plot(xvals, V(xvals), color='black')

    ax.axhline(V_th, linewidth=1, linestyle='dashed', color='black', alpha=0.6)

    input_arrow_height = 0.17
    arrow_head_width = 2.47
    arrow_head_length = 0.04
    for spk in [t_i1, t_i2]:
        ax.arrow(spk, ylim[0], 0, input_arrow_height,
                 color="black",
                 head_width=arrow_head_width,
                 head_length=arrow_head_length,
                 length_includes_head=True,
                 zorder=-1)

    # output spike
    if should_spike:
        t_out = 34.5
        ax.arrow(t_out, ylim[1], 0, -input_arrow_height,
                 color="black",
                 head_width=arrow_head_width,
                 head_length=arrow_head_length,
                 length_includes_head=True,
                 zorder=-1)

    ax.set_yticklabels([])
    if yAnnotated:
        ax.set_ylabel("membrane voltage")
        ax.yaxis.set_label_coords(-0.2, 1.05)

    ax.set_yticks([V_th, 0])
    ax.set_yticklabels([name_of_vth, name_of_vleak])
    ax.set_ylim(ylim)

    if xAnnotated:
        ax.set_xlabel("time [a. u.]")
        ax.xaxis.set_label_coords(.5, -0.15)
    ax.set_xticks([])


def plot_psp_shapes(ax, delay=False, xlabel=False):
    # make the axis
    core.show_axis(ax)
    core.make_spines(ax)

    xvals = np.linspace(0., 100., 200)
    c_m = 0.2
    t_s = 10.
    w = 0.01
    t_i = 15. if not delay else 25

    def theta(x):
        return x > 0

    def V(t, t_m):
        factor = 1.
        if t_m < t_s:
            factor = 6. / t_m

        t[t < t_i] = t_i
        if t_m != t_s:
            ret_val = factor * t_m * t_s / (t_m - t_s) * theta(t - t_i) * \
                (np.exp(-(t - t_i) / t_m) - np.exp(-(t - t_i) / t_s))
        else:
            ret_val = factor * theta(t - t_i) * np.exp(-(t - t_i) / t_m) * (t - t_i)

        ret_val[t < t_i] = 0.
        return ret_val

    taums = [10, ] # [100000000., 20, 10, 0.0001]
    taums_name = [r"\rightarrow \infty", "= 2", "= 1", r"\rightarrow 0"]
    colours = ['C7', 'C8', 'C9', 'C6']
    for t_m, t_m_name, col in zip(taums, taums_name, colours):
        # ax.set_xlabel(r'$\tau_\mathrm{m}$ [ms]')
        lab = "$" + r'\tau_\mathrm{{m}} / \tau_\mathrm{{s}} {}'.format(t_m_name) + "$"
        ax.plot(xvals, V(xvals.copy(), t_m), color=col, label=lab,
                ls='-' if delay else '--')

    ax.set_yticks([])
    ax.set_xticks([])

    ax.set_ylabel("PSP [a. u.]")
    # ax.yaxis.set_label_coords(-0.03, 0.5)

    if xlabel:
        ax.set_xlabel("time [a. u.]")
        ax.xaxis.set_label_coords(.5, -0.075)

    # ax.legend(frameon=False)


def plot_psp_shapes_delay(ax):
    plot_psp_shapes(ax, delay=True, xlabel=True)
    plot_psp_shapes(ax, delay=False, xlabel=True)
    x = 16
    y = 0.5
    dx = 9.
    dy = 0.
    ax.arrow(x, y, dx, dy, length_includes_head=False, color='C3', head_width=0.15, head_length=0.3, linewidth=1.5, zorder=5)
    ax.arrow(x + dx, y + dy, -dx, -dy, length_includes_head=False, color='C3', head_width=0.15, head_length=0.3, linewidth=1.5, zorder=5)
    ax.text(19.5, y * 1.4, "$d$", color='C3')
    x = 35
    y = 0.25
    dx = 0.
    dy = 3.2
    ax.arrow(x, y, dx, dy, length_includes_head=False, color='C3', head_width=0.8, head_length=0.15, linewidth=1.5, zorder=5)
    ax.arrow(x + dx, y + dy, -dx, -dy, length_includes_head=False, color='C3', head_width=0.8, head_length=0.15, linewidth=1.5, zorder=5)
    ax.text(30.5, 1.6, "$w$", color='C3')


def theta(x):
    return x > 0


def psp(t, t_i, t_s, t_m, w, offsets, v_th=None):
    t[t < t_i] = t_i
    if t_m != t_s:
        ret_val = w * t_m * t_s / (t_m - t_s) * theta(t - t_i) * \
            (np.exp(-(t - t_i) / t_m) - np.exp(-(t - t_i) / t_s))
    else:
        ret_val = w * theta(t - t_i) * np.exp(-(t - t_i) / t_m) * (t - t_i)

    ret_val[t < t_i] = 0.
    ret_val = ret_val + offsets
    if v_th is None:
        return ret_val
    else:
        did_spike = np.where(ret_val > v_th, 1., 0.)
        if sum(did_spike):
            spike_idx = np.argmax(did_spike)
            ret_val[spike_idx:] = 0
        return ret_val
