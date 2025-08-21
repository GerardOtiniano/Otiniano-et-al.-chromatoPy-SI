import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from data_processing.load_data import *

df = load_chraomtopy_and_manual(ignore_misidentified = True)
# print(list(df['Sample Name'].unique()))
# brgdgt_types = ["Ia", "IIa","IIa'" , "IIIa", "IIIa'",
#                 "Ib", "IIb", "IIb'", "IIIb", "IIIb'",
#                 "Ic", "IIc", "IIc'", "IIIc", "IIIc'"]
# temp = df[df['variable'].isin(brgdgt_types)]
# df = df[~df['Sample Name'].isin(temp.loc[(temp['chromatopy_pa']==0)&(temp['hand_fa']>0)]['Sample Name'].unique())]
# df = df[~df['Sample Name'].isin(df.loc[(df['chromatopy_pa']>0)&(df['hand_fa']==0)]['Sample Name'].unique())]
# print(list(df['Sample Name'].unique()))

# sample_to_ignore = ['H2202081', 'H2308012', 'H2202085', 'H2202087', 'H1608000014', 'H1608000013', 'H2307071']
# df = df[~df['Sample Name'].isin(sample_to_ignore)]
brgdgt_types = ["Ia", "IIa","IIa'" , "IIIa", "IIIa'",
                "Ib", "IIb", "IIb'", "IIIb", "IIIb'",
                "Ic", "IIc", "IIc'", "IIIc", "IIIc'"]
df_wide = df.pivot(index='Sample Name', columns='variable', values='chromatopy_fa').reset_index()
df_wide['MBT_5Me'] = (
    (df_wide['Ia'] + df_wide['Ib'] + df_wide['Ic']) /
    (df_wide['Ia'] + df_wide['Ib'] + df_wide['Ic'] +
     df_wide['IIa'] + df_wide['IIb'] + df_wide['IIc'] +
     df_wide['IIIa'] + df_wide['IIIb'] + df_wide['IIIc']))


df_ind = df_wide[['Sample Name', 'MBT_5Me']]

df_wide = df.pivot(index='Sample Name', columns='variable', values='hand_fa').reset_index()
df_wide['MBT_5Me_hand'] = (
    (df_wide['Ia'] + df_wide['Ib'] + df_wide['Ic']) /
    (df_wide['Ia'] + df_wide['Ib'] + df_wide['Ic'] +
     df_wide['IIa']  + df_wide['IIb'] + df_wide['IIc'] + 
     df_wide['IIIa'] +  df_wide['IIIb'] + df_wide['IIIc']))


# fig, ax = plt.subplots()#figsize=(5,5))
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

# histogram
ax_hist.hist(df_ind['mbt_diff'], bins=20, density=False,
             color='gray', alpha=0.6)
ax_kde = ax_hist.twinx()
# Smooth KDE
kde = gaussian_kde(df_ind['mbt_diff'])
xgrid = np.linspace(df_ind['mbt_diff'].min(), df_ind['mbt_diff'].max(), 300)
ax_kde.plot(xgrid, kde(xgrid), lw=1.5, c='k')
ax_hist.axvline(0, c='k', linestyle='--')
ax_hist.tick_params(labelbottom=False, left=False, right=False)
ax_hist.set_ylabel("Sample Count", labelpad=20)
ax_kde.set_ylabel("Density")

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
plt.savefig("Fig. 4.png", dpi=300)
plt.show()

print(f"median: {np.nanmedian(df_ind['mbt_diff'])}")
print(f"2.5%: {np.percentile(df_ind['mbt_diff'], 2.5)}")
print(f"97.5%: {np.percentile(df_ind['mbt_diff'], 97.5)}")

stat, p_value = wilcoxon(df_ind['mbt_diff'])
print(stat)
print(p_value)
