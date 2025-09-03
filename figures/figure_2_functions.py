import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch

def sigma_avg_from_asym_ci95(lower_ci, upper_ci):
    """
    Symmetrize asymmetric 95% CI half-widths, then convert to 1σ.
    Returns scalar/Series: ((upper_ci + lower_ci)/2) / 1.96
    """
    return ( (upper_ci + lower_ci) / 2.0 ) / 1.96

def user_comparison(df):
    sub = df[(df['chromatopy_ra'] != 0) & (df['user_2_ra'] != 0)].copy()

    A = sub["chromatopy_pa"].astype(float)
    B = sub["user_2_pa"].astype(float)
    diff = A - B
    sub["diff"] = diff.abs() 
    denom = (A + B) / 2.0
    sub["perc_diff"] = np.where(denom != 0, (diff / denom) * 100.0, np.nan)
    A_lo = sub["chromatopy_pa_lower"].astype(float)
    A_hi = sub["chromatopy_pa_upper"].astype(float)
    B_lo = sub["user_2_pa_lower"].astype(float)
    B_hi = sub["user_2_pa_upper"].astype(float)

    sigA = sigma_avg_from_asym_ci95(A_lo, A_hi)
    sigB = sigma_avg_from_asym_ci95(B_lo, B_hi)
    sub["err"] = np.sqrt(sigA**2 + sigB**2)  # combined σ for A−B
    sub['rel_err'] = (np.abs(sub["err"])/(sub["chromatopy_pa"]))*100
    return sub

def threshold_outliers(results):
    clean_df = results["clean_df"]
    thresh = np.nanpercentile(clean_df.rel_err, 90)
    below = clean_df[clean_df.rel_err<thresh]
    above = clean_df[clean_df.rel_err>thresh]
    return thresh, below, above

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

def remove_samples(df, ignore_isogdgts, ignore_brgdgts):
    brGDGTs = ["Ia", "IIa", "IIa'", "IIIa","IIIa'",
               "Ib", "IIb", "IIb'", "IIIb","IIIb'",
               "Ic", "IIc", "IIc'", "IIIc","IIIc'",]
    isoGDGTs = ["GDGT-0", "GDGT-1", "GDGT-2", "GDGT-3", "GDGT-4", "GDGT-4'"]
    ignored = df.loc[
        ((df['Sample Name'].isin(ignore_brgdgts)) & (df['variable'].isin(brGDGTs)))|
        ((df['Sample Name'].isin(ignore_isogdgts)) & (df['variable'].isin(isoGDGTs)))]
    df = df.loc[~((df['Sample Name'].isin(ignore_isogdgts)) & (df['variable'].isin(isoGDGTs)))]
    retained = df.loc[~((df['Sample Name'].isin(ignore_brgdgts)) & (df['variable'].isin(brGDGTs)))]
    return retained, ignored