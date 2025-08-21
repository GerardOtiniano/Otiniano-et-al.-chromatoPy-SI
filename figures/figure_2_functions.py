import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch

def bland_altman_metrics(df):
    """
    Return the Series/metrics needed for a Bland-Altman plot.
    Handles NaNs, zeros, and keeps asymmetric uncertainties intact.
    """
    # Ensure numeric and clean
    cols_to_check = ["chromatopy_ra", "hand_ra",
                     "chromatopy_pa", "chromatopy_pa_upper", "chromatopy_pa_lower",
                     "chromatopy_ra_upper", "chromatopy_ra_lower"]
    for col in cols_to_check:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows where either analyst has missing or zero relative abundance
    keep = (df["chromatopy_ra"] > 0) & (df["hand_ra"] > 0)
    d = df.loc[keep].copy()

    # Pairwise stats
    diff = d["chromatopy_ra"] - d["hand_ra"]
    mean = d[["chromatopy_ra", "hand_ra"]].mean(axis=1)

    # Relative % error in peak area (symmetric assumption for plotting)
    d['rel_err'] = ((np.abs(d["chromatopy_pa_upper"]) + np.abs(d["chromatopy_pa_lower"])) / (2 * d["chromatopy_pa"]))*100
    return d

def user_comparison(df):
    """
    Build DataFrames that compare ChromatoPy peak area
    to a second user's peak area.

    * Rows where one analyst reports zero when the other reports >0 are dropped.
    * Rows with mis-identified samples are excluded.
    """
    # Remove missed peaks to avoid NaN values
    sub = df[df['chromatopy_ra']!=0]
    sub = sub[sub['user_2_ra']!=0]

    # difference between ChromatoPy and user 2 
    """
    # Two samples with effectively 0 peak area (H1801000128 IIIc` has a 
    peak area of 0.838 and H1801000191 GDGT-3 has a peak area of 1.444) are
    removed from Figure 2 panels A, B, and C for clarity.
    """
    sub['diff'] = np.abs(((sub["chromatopy_pa"] - sub["user_2_pa"])))
    numerator = (sub["chromatopy_pa"] - sub["user_2_pa"])
    denominator =(sub["chromatopy_pa"]+sub["user_2_pa"])/2
    sub['perc_diff'] = (numerator / denominator)  * 100

    # Error from both users
    err_1 = (np.abs(sub["chromatopy_pa_upper"])+np.abs(sub["chromatopy_pa_lower"]))/2
    err_2 = (np.abs(sub["user_2_pa_upper"])+np.abs(sub["user_2_pa_lower"]))/2
    sub['err'] = np.sqrt(err_1**2+err_2**2)
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
    
def add_z_uncertainty_flag(df, diff_col: str, uncert_col: str, 
                           clip_small_unc: float = 1e-12,new_col: str = "z_uncertainty"):
    d = df.copy()
    sigma = d[uncert_col].abs().clip(lower=clip_small_unc)
    z = d[diff_col] / sigma
    d[new_col] = (z.abs() > 1).astype(int)
    return d

def zscore_summary(
    df,
    diff_col,             
    uncert_col,            
    split_col,         
    threshold=1000,
    min_group_n=1):
    d = df[[diff_col, uncert_col, split_col]].copy()

    sigma = d[uncert_col].abs()
    d['z'] = d[diff_col] / sigma
    d['abs_z'] = d['z'].abs()

    lo = d[d[split_col] <= threshold].copy()
    hi = d[d[split_col] >  threshold].copy()

    def _summ(group):
        n = len(group)
        if n < min_group_n:
            return {
                'n': n,
                'median_abs_z': np.nan,
                'mean_abs_z': np.nan,
                'p_|z|<=1': np.nan,
                'p_|z|<=2': np.nan,
                'p_|z|<=3': np.nan,
                'median_z': np.nan,
                'mean_z': np.nan,
                'q_abs_z_2.5': np.nan,
                'q_abs_z_97.5': np.nan}
        return {
            'n': n,
            'median_abs_z': float(np.median(group['abs_z'])),
            'mean_abs_z': float(np.mean(group['abs_z'])),
            'p_|z|<=1': float((group['abs_z'] <= 1).mean()),
            'p_|z|<=2': float((group['abs_z'] <= 2).mean()),
            'p_|z|<=3': float((group['abs_z'] <= 3).mean()),
            'median_z': float(np.median(group['z'])),
            'mean_z': float(np.mean(group['z'])),
            'q_abs_z_2.5': float(np.percentile(group['abs_z'], 2.5)),
            'q_abs_z_97.5': float(np.percentile(group['abs_z'], 97.5)),}
    results = {
        'threshold': threshold,
        'below_or_equal': _summ(lo),
        'above': _summ(hi)}
    return results