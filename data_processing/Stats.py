import matplotlib.pyplot as plt
from data_processing.load_data import *
from figures.figure_1_functions import *
from figures.figure_2_functions import user_comparison
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon
from scipy.stats import ks_2samp
from figures.figure_2_functions import *

# %% Import data
df = load_chraomtopy_and_manual()

sub1 = (df['chromatopy_ra'] == 0) & (df['hand_ra'] > 0)
sub2 = (df['chromatopy_ra'] > 0) & (df['hand_ra'] == 0)
drop_mask = sub1 | sub2

# General comparisons
print(f"Detected manually but not by chromatoPy: {int(sub1.sum())}")
print(f"Detected by chromatoPy but not manually: {int(sub2.sum())}")
df = df.loc[~drop_mask].copy()
print(f"Number of commonly identified peaks: {len(df)}")

rpd_pct = 100.0 * (df['chromatopy_pa'] - df['user_2_pa']) / ((df['chromatopy_pa'] + df['user_2_pa']) / 2.0)
median = float(np.nanmedian(rpd_pct))
p16, p84 = np.nanpercentile(rpd_pct, [15.865, 84.135])

print(f"Median percent difference: {median:.3f}%")
print(f"Approx. -1σ: {p16:.3f}%")
print(f"Approx. +1σ: {p84:.3f}%")
print(f"Approx. σ (half-width): {(p84 - p16)/2:.3f}%")

# %% Chromatopy - uncertainty test 
thresh = 500
print(np.mean(sub[sub['chromatopy_pa']<thresh]['perc_diff']))
print(np.percentile(sub[sub['chromatopy_pa']<thresh]['perc_diff'], 2.5))
print(np.percentile(sub[sub['chromatopy_pa']<thresh]['perc_diff'], 97.5))
print(np.mean(sub[sub['chromatopy_pa']>thresh]['perc_diff']))
print(np.percentile(sub[sub['chromatopy_pa']>thresh]['perc_diff'], 2.5))
print(np.percentile(sub[sub['chromatopy_pa']>thresh]['perc_diff'], 97.5))

# %% Z-score uncertainty
df = load_chraomtopy_and_manual()

# Remove misidentified samples
ignore_isogdgts = ['H1608000189', 'H1801000129', 'H1801000194', 'H1801000130']
ignore_brgdgts = ['H1608000014', 'H2202085', 'H2202081', 'H2202087', 'H1608000013', 'H2305015', 'H1805000004', 'H2307064', 'H2204051', 'H1801000131']
df, ignored = remove_samples(df, ignore_isogdgts, ignore_brgdgts)
results = bland_altman_metrics(df)
sub = user_comparison(df)
sub['uncertainty'] = np.sqrt(((sub['chromatopy_pa_lower']+sub['chromatopy_pa_upper'])/2)**2+((sub['user_2_pa_lower']+sub['user_2_pa_upper'])/2)**2)

res = zscore_summary(
    sub,
    diff_col='perc_diff',
    uncert_col='uncertainty',
    split_col='chromatopy_pa',
    threshold=500)
res

# %% Chromatopy-Manual comarison

brgdgts = [ "Ia", "Ib", "Ic", "IIa", "IIb", "IIc", "IIIa", "IIIb", "IIIc",
 "IIa'", "IIb'", "IIc'", "IIIa'", "IIIb'", "IIIc'"]
isogdgts = ['GDGT-0', 'GDGT-1', 'GDGT-2', 'GDGT-3', 'GDGT-4', "GDGT-4'"]

#  Wilcoxon difference in medians between of isoGDGTs and brGDGTs
chromatopy = 'chromatopy_fa'
manual = 'hand_fa'
sub1 = df[chromatopy] == 0
sub2 = df[manual] == 0
drop_mask = sub1 | sub2
df = df[~drop_mask] # Only consider detected peaks
for gdgt, g_set in zip(["isogdgts", "brgdgts"],[isogdgts, brgdgts]):
    for val in ["ra", "fa"]: 
        temp = df[df['variable'].isin(g_set)]
        temp['difference'] = df[f"chromatopy_{val}"] - df[f"hand_{val}"]
        stat, p_value = wilcoxon(temp['difference'])
        print(f"Wilcoxon signed rank test for {gdgt}:\nstat = {stat}\np = {p_value}\n median = {np.nanmedian(temp['difference'])}")

# %% Wilcoxon test between relpicates
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

variables = ["Ia", "Ib", "Ic", "IIa", "IIb", "IIc", "IIIa", "IIIb", "IIIc",
    "IIa'", "IIb'", "IIc'", "IIIa'", "IIIb'", "IIIc'",
    'GDGT-0', 'GDGT-1', 'GDGT-2', 'GDGT-3', 'GDGT-4', "GDGT-4'"]

def build_table(df, x_col, y_col, peak_type):
    results = []
    for var in variables:
        
        temp = df[df['variable'] == var].copy()
        temp = temp.dropna()
        temp = temp[temp['chromatopy_pa']>0]
        temp["difference"] = temp[x_col] - temp[y_col]
        
        # Wilcoxon test
        stat, p_value = wilcoxon(temp['difference'])
        
        # Median absolute difference
        med_diff = np.median(temp['difference'])
        
        # Percent median difference (relative to median of x_col)
        med_peak = np.nanmedian(temp[x_col])
        percent_diff = (med_diff / med_peak) * 100 if med_peak != 0 else np.nan
        median_val = np.median(temp[x_col])
        temp["temporary_uncertainty"] = (((temp["chromatopy_pa_lower"]+temp["chromatopy_pa_upper"])/2)/temp['chromatopy_pa'])*temp['chromatopy_ra']
        if peak_type == 'fa':
            percent_uncertainty = np.median((temp["chromatopy_fa_combined_error"]/temp[x_col])*100)
        else:
            percent_uncertainty = np.median((temp["temporary_uncertainty"]/temp[x_col])*100)
        results.append({
            "GDGT": var,
            "wilcox": stat,
            "p_value": p_value,
            "median_diff": med_diff,
            "percent_diff": percent_diff,
            "percent_uncertainty": percent_uncertainty,
            "median_val": median_val,
            "number of peaks": len(temp[x_col])
        })
    
    # Formatting
    df_out = pd.DataFrame(results)
    
    # Format p-values
    df_out["p_value"] = df_out["p_value"].apply(lambda p: "<0.001" if p < 0.001 else f"{p:.3f}")
    
    # Format median difference in scientific notation
    df_out["median_diff"] = df_out["median_diff"].apply(lambda x: f"{x:.3e}")
    
    # Round percent diff to 2 decimals
    df_out["median_val"] = df_out["median_val"].apply(lambda p: "<0.001" if p < 0.001 else f"{p:.3f}")
    df_out["percent_diff"] = df_out["percent_diff"].round(3)
    df_out["percent_uncertainty"] = df_out["percent_uncertainty"].round(3)
    
    return df_out

# Build both tables
table_ra = build_table(df, "chromatopy_ra", "hand_ra", "ra")
table_fa = build_table(df, "chromatopy_fa", "hand_fa", "fa")

# Display
print("\n=== Table: Chromatopy RA vs Manual RA ===")
print(table_ra.to_string(index=False))

print("\n=== Table: Chromatopy FA vs Manual FA ===")
print(table_fa.to_string(index=False))