#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
#  File:        make_stap_figs_v4.py
#  Project:     Symmetry-Group Covariance Estimation for Adaptive Arrays
#  Brief:       Renders the three result figures.
#
#  Purpose
#    Draws one single-column figure per result, sized for a two-column
#    manuscript, from the committed result files.
#
#  Description
#    fig_matched  SINR loss against training support under exact symmetry.
#    fig_scaling  snapshots to 3 dB against aperture for both symmetry
#                 families, with the 2M/d_eff prediction overlaid.
#    fig_calib    SINR loss under broken symmetry for both shrinkage
#                 calibrations, the projection and the achievable envelope.
#    Panels are drawn from different runs on purpose, since they answer
#    different questions; no curve appears in more than one figure.
#
#  Inputs
#    stap_sinr.json, stap_scaling.json, nll_calib.json
#
#  Outputs
#    figures/fig_matched.{pdf,png}, fig_scaling.{pdf,png},
#    figures/fig_calib.{pdf,png}
#
#  Usage
#    python3 make_stap_figs_v4.py
#
#  Version
#    Created:       bundle v7, 2026-09-01
#    Last modified: bundle v7.6, 2026-09-02
#
#  Revision history
#    v7     2026-09-01   created; three single-column figures replace the
#                        three-panel spanning figure of v6
#    v7.1   2026-09-01   source headers added
#    v7.4   2026-09-02   explicit log-axis ticks, larger axis fonts and
#                        headroom for the legends
#    v7.5   2026-09-02   legends moved above the axes, outside the frame,
#                        uniformly across the three panels; panel height
#                        2.50 in
#    v7.6   2026-09-02   figures saved at exactly the requested size
#                        under a constrained layout, so the aspect ratio
#                        no longer depends on how the installed library
#                        measures the legend
#
#  Author
#    Mitchell A. Thornton  <mitch@smu.edu>
#    ORCID 0000-0003-3559-9511
#
#  License
#    SPDX-License-Identifier: MIT
#    The MIT license grants copyright permissions only and grants no
#    rights under patents. See PATENTS.md.
#
#  Copyright (c) 2026 Mitchell A. Thornton
# ============================================================================

"""The three result figures.

Each is a single-column figure, one per result.

  figures/fig_matched.pdf   matched symmetry: SINR loss versus training support
  figures/fig_scaling.pdf   sample complexity versus aperture
  figures/fig_calib.pdf     broken symmetry: the two shrinkage calibrations

Reads stap_sinr.json, nll_calib.json, stap_scaling.json.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# single IEEEtran conference column is about 3.5 in
FIGSIZE = (3.45, 2.50)
plt.rcParams.update({
    "font.size": 8.0,
    "axes.titlesize": 8,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 6.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.4,
})

sinr = json.load(open("stap_sinr.json"))
nll = json.load(open("nll_calib.json"))
scal = json.load(open("stap_scaling.json"))
MA = sinr["matched"]
NM = nll["mismatch"]

os.makedirs("figures", exist_ok=True)


def save(fig, name):
    # The constrained layout engine set at figure creation does the fitting, so
    # there is no tight_layout call here; calling one would switch the engine
    # and undo the room reserved for the legend.
    # No tight bounding box. With the legend outside the axes a tight box makes
    # the saved aspect ratio depend on how the installed matplotlib measures the
    # legend, and the document scales each figure to the column width, so that
    # difference becomes page layout. Saving at exactly FIGSIZE makes the height
    # the document gives a figure the same on every machine.
    w, h = fig.get_size_inches()
    assert (round(w, 3), round(h, 3)) == FIGSIZE, (w, h)
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/{name}.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote figures/{name}.pdf")


def outside_legend(ax, ncol):
    """Place the legend above the axes, outside the frame.

    An in-frame legend on these panels covers curve content, and on the
    matched-symmetry and calibration panels there is no corner it can be moved
    to that does not. The legend is attached to the figure with an outside
    location, which the constrained layout engine reserves room for inside the
    fixed figure size, rather than to the axes with an anchor that would push
    outside the figure and have to be recovered by a tight bounding box.
    """
    fig = ax.get_figure()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=ncol,
               frameon=False, columnspacing=1.1, handlelength=1.7,
               handletextpad=0.5, borderpad=0.2, labelspacing=0.25)


def log_x_ticks(ax, ticks):
    """Label a log x-axis at the given values only.

    Matplotlib labels minor decades as well when a log axis spans a little
    over one decade, and at this figure width those labels collide. Fixing
    the major ticks and dropping the minor ones keeps the axis readable.
    """
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(t) for t in ticks]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())


def log_y_ticks(ax, ticks):
    """Label a log y-axis at the given values only."""
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(mticker.FixedFormatter([str(t) for t in ticks]))
    ax.yaxis.set_minor_locator(mticker.NullLocator())


# ------------------------------------------------------------ Fig. 1: matched
fig, ax = plt.subplots(figsize=FIGSIZE, layout='constrained')
for key, lab, mk, col in [('scm', 'sample (SCM)', 's--', '#555555'),
                          ('dl', 'loaded SCM (oracle level)', 'v--', '#2ca02c'),
                          ('persym', 'persymmetric ($Z_2$)', '^--', '#ff7f0e'),
                          ('group', 'group average', 'o-', '#1f77b4'),
                          ('shrink', 'shrinkage (Frobenius)', 'D-', '#d62728')]:
    y = [np.nan if v is None else v for v in MA[key]]
    ax.plot(MA["Ks"], y, mk, color=col, label=lab)
ax.axvline(MA["M"], color='k', ls=':', lw=0.8)
ax.text(MA["M"] * 1.06, -13.4, '$K=M$', fontsize=6.5)
ax.set_xscale('log')
ax.set_ylim(-14, 1)
ax.set_xlabel("training snapshots $K$")
ax.set_ylabel("mean SINR loss (dB)")
ax.grid(alpha=0.3, which='both')
log_x_ticks(ax, [2, 4, 8, 16, 32, 64])
outside_legend(ax, 2)
save(fig, "fig_matched")

# ------------------------------------------------------- Fig. 2: calibration
fig, ax = plt.subplots(figsize=FIGSIZE, layout='constrained')
for key, lab, mk, col in [('lsmi', 'loaded SCM (oracle level)', 'v--', '#2ca02c'),
                          ('group', 'projection ($\\alpha=1$)', 'o-', '#1f77b4'),
                          ('frob', 'shrinkage (Frobenius)', 'D-', '#d62728'),
                          ('nll', 'shrinkage (held-out NLL)', 'P-', '#9467bd')]:
    y = [np.nan if v is None else v for v in NM[key]]
    ax.plot(NM["Ks"], y, mk, color=col, label=lab)
ax.plot(NM["Ks"], [np.nan if v is None else v for v in NM['oracle']], ':',
        color='#777777', lw=1.4, label='best $\\alpha$ (oracle)')
ax.axvline(NM["M"], color='k', ls=':', lw=0.8)
ax.set_xscale('log')
ax.set_ylim(-8.2, 0.5)
ax.set_xlabel("training snapshots $K$")
ax.set_ylabel("mean SINR loss (dB)")
ax.grid(alpha=0.3, which='both')
log_x_ticks(ax, [4, 8, 16, 32, 64])
outside_legend(ax, 2)
save(fig, "fig_calib")

# ----------------------------------------------------------- Fig. 3: scaling
fig, ax = plt.subplots(figsize=FIGSIZE, layout='constrained')
Ms = [r["M"] for r in scal["growing"]]
gro, fix = scal["growing"], scal["fixed"]
ax.plot(Ms, [r["k3_scm"] for r in gro], 's--', color='#555555',
        label='sample ($\\approx 2M$)')
ax.plot(Ms, [r["k3_persym"] for r in gro], '^--', color='#ff7f0e',
        label='persymmetric ($\\approx M$)')
ax.plot(Ms, [r["k3_lsmi"] for r in gro], 'v--', color='#2ca02c',
        label='loaded SCM ($\\approx 2J$)')
ax.plot(Ms, [r["k3_group"] for r in fix], 'o--', color='#7fbfe8',
        label='group, fixed $J=4$')
ax.plot(Ms, [r["k3_group"] for r in gro], 'o-', color='#1f77b4',
        label='group, growing $J=M/4$', ms=4.2, lw=1.6)
ax.plot(Ms, [2 * r["M"] / r["deff"] for r in gro], 'k:', lw=1.3,
        label='$2M/d_{\\rm eff}$')
ax.plot(Ms, [2 * r["M"] / r["deff"] for r in fix], 'k:', lw=1.3)
ax.set_yscale('log')
ax.set_ylim(1.9, 200)
ax.set_xlabel("array size $M$")
ax.set_ylabel("snapshots to 3 dB SINR loss")
ax.grid(alpha=0.3, which='both')
log_y_ticks(ax, [2, 5, 10, 20, 50, 100])
outside_legend(ax, 2)
save(fig, "fig_scaling")
