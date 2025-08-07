import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch


def bland_altman_metrics(df):
    """
    Return the Series/metrics needed for a Bland-Altman plot.
    """
    # Remove undected GDGTs to avoid NaN errors
    keep = (df["chromatopy_ra"] != 0) & (df["hand_ra"] > 0) 
    d = df.loc[keep].copy()

    # Pairwise statistics
    diff  = d["chromatopy_ra"] - d["hand_ra"]
    mean  = d[["chromatopy_ra", "hand_ra"]].mean(axis=1)

    # % errors 
    pos_err = 100 * d["chromatopy_pa_upper"] / d["chromatopy_pa"]
    neg_err = 100 * d["chromatopy_pa_lower"] / d["chromatopy_pa"]
    sym_err = (100 / d["chromatopy_ra_lower"] * ((d["chromatopy_ra_upper"] + 
              d["chromatopy_ra"]) - (d["chromatopy_ra"] - d["chromatopy_ra_lower"]))/ 2)

    # Bias & 95 % limits of agreement
    bias     = diff.mean()
    std_diff = diff.std(ddof=1)
    loa      = bias + np.array([-1.96, 1.96]) * std_diff   # [lower, upper]

    return {"clean_df" : d, "mean" : mean,"diff" : diff,"pos_err" : pos_err,
        "neg_err" : neg_err, "sym_err" : sym_err, "bias" : bias,
        "loa_lower" : loa[0], "loa_upper" : loa[1]}


def user_comparison(df):
    """
    Build DataFrames that compare ChromatoPy peak area
    to a second user's peak area.

    * Rows where one analyst reports zero when the other reports >0 are dropped.
    * Rows with mis-identified samples are excluded.
    * Differences below `diff_floor` are discarded.
    """
    keep = ~(
        ((df["chromatopy_ra"] == 0) & (df["hand_ra"] > 0))
        | ((df["chromatopy_ra"] > 0) & (df["hand_ra"] == 0)))
    sub = df.loc[keep].fillna(0).copy()

    # % difference between ChromatoPy and user 2
    sub["diff"] = (sub["chromatopy_pa"] - sub["user_2_pa"]) / sub["chromatopy_pa"] * 100
    """
    # Two samples with effectively 0 peak area (H1801000128 IIIc` has a 
    peak area of 0.838 and H1801000191 GDGT-3 has a peak area of 1.444) are
    removed from Figure 2 panels A, B, and C for clarity.
    """
    sub = sub[sub['diff']>-1000] 

    # Symmetric absolute error (upper+lower)
    sub["err"] = sub["chromatopy_pa_upper"] + sub["chromatopy_pa_lower"]
    thresh     = np.sqrt(2) * sub["err"]
    smaller = sub[np.abs(sub["diff"]) < thresh]
    larger  = sub[np.abs(sub["diff"]) > thresh]
    return smaller, larger

def threshold_outliers(results):
    rel_uncertainty = results["pos_err"]
    thresh = np.mean(rel_uncertainty) + 2 * np.std(rel_uncertainty)
    clean_df = results["clean_df"]
    below = clean_df[rel_uncertainty < thresh]
    above = clean_df[rel_uncertainty >= thresh]
    return below, above, thresh

def plot_connections(axs):
    ax1_xmin, ax1_xmax = axs[1].get_xlim()
    ax1_ymin, ax1_ymax = axs[1].get_ylim()
    ax0_xmin, ax0_xmax = axs[0].get_xlim()
    ax0_ymin, ax0_ymax = axs[0].get_ylim()
    axs[1].axvline(0, c='grey', linestyle='--', zorder=0)
    axs[0].axvline(0, c='grey', linestyle='--', zorder=0)
    axs[1].axvline(ax1_xmax, c='grey', linestyle='--', zorder=0)
    axs[0].axvline(ax1_xmax, c='grey', linestyle='--', zorder=0)
    for x in [0, ax1_xmax]:
        con = ConnectionPatch(xyA=(x, ax1_ymax), xyB=(x, ax0_ymin), coordsA="data", coordsB="data",
                              axesA=axs[1], axesB=axs[0], color="grey", linestyle='--', zorder=0)
        axs[1].add_artist(con)

def connect_vertical_patch(ax_from, ax_to, x_val, y_from, y_to, color='grey', linestyle='--', zorder=0):
    con = ConnectionPatch(
        xyA=(x_val, y_from), xyB=(x_val, y_to),
        coordsA="data", coordsB="data",
        axesA=ax_from, axesB=ax_to,
        color=color, linestyle=linestyle, zorder=zorder)
    ax_from.add_artist(con)

def clean_spines(ax):
    ax.spines[['right', 'top']].set_visible(False)