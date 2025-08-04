"""
Python code for Figure 4.
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde


# Import data
chpy_man_data = pd.read_csv('data/chromatoPy and manual data.csv') # subset of overlapping chromatoPy and manually integrated data results
chpy_full = pd.read_csv('data/chromatopy_data_full.csv') # results from chromatopy integration (no subsetting by manual integrations)
user_2_data = pd.read_csv('data/chromatoPy_user_2_data_peak_area.csv') # chromatoPy integrations from independent second user
user_2_fa = pd.read_csv('data/user_2_data_fractional_abundance.csv')

# Process data
chpy_man_data['chromatopy_ra'].fillna(0, inplace=True)
chpy_man_data['chromatopy_ra'].fillna(0, inplace=True)
chpy_man_data['hand_ra'].fillna(0, inplace=True)
chpy_man_data['hand_ra'].fillna(0, inplace=True)

columns_of_interest = ["Sample Name", "Ia", "Ib", "Ic", "IIa", "IIb", "IIc", "IIIa", "IIIb", "IIIc", "IIa'", "IIb'", "IIc'", "IIIa'", "IIIb'", "IIIc'", 'GDGT-0', 'GDGT-1', 'GDGT-2', 'GDGT-3', 'GDGT-4', "GDGT-4'"]
brgdgts = ["Ia", "Ib", "Ic", "IIa", "IIb", "IIc", "IIIa", "IIIb", "IIIc", "IIa'", "IIb'", "IIc'", "IIIa'", "IIIb'", "IIIc'"]
user_2_data = user_2_data[columns_of_interest]
user_2_data = user_2_data.melt(id_vars='Sample Name')
user_2_data = user_2_data.rename(columns={'value': 'user_2_values'})

# Seperate mis-identified samples
misid_names = ['H1608000196', 'H1608000191', 'H1608000189', 'H2102002', 'H2102003', 'H2102004'] # names of samples with misidentified peaks
misidentified_samples = chpy_man_data[chpy_man_data['Sample Name'].isin(misid_names)]
chpy_man_data = chpy_man_data.loc[~((chpy_man_data['Sample Name'].isin(misid_names))&(chpy_man_data['variable'].isin(brgdgts)))]

user_peak_areas=pd.merge(user_2_data, chpy_full[['Sample Name', 'variable', 'chromatopy_value']],on=['Sample Name', 'variable'], how='right')
df = pd.merge(chpy_man_data, user_peak_areas, on=['Sample Name', 'variable'], how='left', suffixes=("","_PA"))
df=df.fillna(0)

df = df.loc[~((df['Sample Name'].isin(misidentified_samples))&(df['variable'].isin(brgdgts)))]
df_wide = df.pivot(index='Sample Name', columns='variable', values='chromatopy_fa').reset_index()

# MBT'5Me for chromatopy data
df_wide['MBT_5Me'] = (
    (df_wide['Ia'] + df_wide['Ib'] + df_wide['Ic']) /
    (df_wide['Ia'] + df_wide['Ib'] + df_wide['Ic'] +
     df_wide['IIa'] + df_wide['IIb'] + df_wide['IIc'] +
     df_wide['IIIa'] + df_wide['IIIb'] + df_wide['IIIc']))

# MBT'5Me for manual integration data
df_ind = df_wide[['Sample Name', 'MBT_5Me']]
df_wide = df.pivot(index='Sample Name', columns='variable', values='hand_fa').reset_index()
df_wide['MBT_5Me_hand'] = (
    (df_wide['Ia'] + df_wide['Ib'] + df_wide['Ic']) /
    (df_wide['Ia'] + df_wide['Ib'] + df_wide['Ic'] +
     df_wide['IIa']  + df_wide['IIb'] + df_wide['IIc'] + 
     df_wide['IIIa'] +  df_wide['IIIb'] + df_wide['IIIc']))


fig = plt.figure(figsize=(7, 8), constrained_layout=False)
gs  = gridspec.GridSpec(
    nrows=2, ncols=1,
    height_ratios=[4, 1],   # top:bottom = 1:4
    hspace=0.1)
fig.subplots_adjust(left=0.2) 
ax_hist = fig.add_subplot(gs[0])
ax      = fig.add_subplot(gs[1], sharex=ax_hist)
df_ind['mbt_diff'] = df_ind['MBT_5Me'] - df_wide['MBT_5Me_hand']
df_ind = df_ind[~df_ind['mbt_diff'].isna()]

for xpos, ypos in zip([4.7/40.01, 2.47/32.42, 4.8/31.45, 1.16/14.75],[-1, -2, -3, -4]):
    ax.plot([-xpos, xpos], [ypos, ypos], c='red')
ax.axvline(0, c='k', linestyle='--')

ax.set_xlabel(r"$\Delta$ MBT$^{\prime}_{5Me}$")
ypos_label = ['Naafs et al.\n(2017)', 'Russell et al.\n(2018)', 'De Jonge et al.\n(2014)', 'Otiniano et al.\n(2023)' ]
ypos = [-1,-2,-3,-4]
all_locs    = ypos
all_labs    = ypos_label
ax.set_yticks(all_locs)
ax.set_yticklabels(all_labs)

divider = make_axes_locatable(ax)
# Histogram
ax_hist.hist(df_ind['mbt_diff'], bins=20, density=False,
             color='gray', alpha=0.6)
# KDE
ax_kde = ax_hist.twinx()
kde = gaussian_kde(df_ind['mbt_diff'])
xgrid = np.linspace(df_ind['mbt_diff'].min(), df_ind['mbt_diff'].max(), 300)
ax_kde.plot(xgrid, kde(xgrid), lw=1.5, c='k')
ax_hist.axvline(0, c='k', linestyle='--')


ax_hist.tick_params(labelbottom=False, left=False, right=False)
ax_hist.set_ylabel("Sample Count", labelpad=20)
ax_kde.set_ylabel("Density")# ax_kde.set_ylim(0,55)

ax_hist.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=True,
    labelbottom=False,
    labeltop=False)
ax.text(0.02, 0.84, '(B)',
     horizontalalignment='left',
     verticalalignment='center',
     transform = ax.transAxes)
ax_hist.text(0.02, 0.96, '(A)',
     horizontalalignment='left',
     verticalalignment='center',
     transform = ax_hist.transAxes)
plt.show()