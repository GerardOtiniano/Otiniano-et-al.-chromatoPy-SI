"""
Python code for Figure 2.
"""
# Import packages
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.patches import ConnectionPatch
import numpy as np
# Import data
chpy_man_data = pd.read_csv('data/chromatoPy and manual data.csv') # subset of overlapping chromatoPy and manually integrated data results
chpy_full = pd.read_csv('data/chromatopy_data_full.csv') # results from chromatopy integration (no subsetting by manual integrations)
user_2_data = pd.read_csv('data/chromatoPy_user_2_data_peak_area.csv') # chromatoPy integrations from independent second user
user_2_fa = pd.read_csv('data/user_2_data_fractional_abundance.csv')

# Process data
chpy_man_data['chromatopy_ra'].fillna(0, inplace=True)
chpy_man_data['chromatopy_fa'].fillna(0, inplace=True)
chpy_man_data['hand_ra'].fillna(0, inplace=True)
chpy_man_data['hand_fa'].fillna(0, inplace=True)

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


zeros = df.loc[(df.chromatopy_ra==0) | (df.hand_ra == 0)]
subset = df.loc[~((df.chromatopy_ra==0) & (df.hand_ra > 0))]
subset = subset.loc[~((subset.chromatopy_ra>0) & (subset.hand_ra == 0))]


legend_map = {} 
plt.rcParams.update({'font.size': 10})
df_nonzero = subset
brgdgt_types = ["Ia", "IIa","IIa'" , "IIIa", "IIIa'",
                "Ib", "IIb", "IIb'", "IIIb", "IIIb'",
                "Ic", "IIc", "IIc'", "IIIc", "IIIc'", 
                'GDGT-0', 'GDGT-1', 'GDGT-2', 'GDGT-3', 'GDGT-4', "GDGT-4'"]

fig, axs = plt.subplots(3, 3, figsize=(15, 7), constrained_layout=True)
ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axs[0,0], axs[1,0], axs[2,0], axs[0,1], axs[1,1], axs[2,1], axs[0,2], axs[1,2], axs[2,2]

df_nonzero = df.loc[~(df.chromatopy_ra==0) & (df.hand_ra > 0)]
diff = df_nonzero.chromatopy_ra - df_nonzero.hand_ra
mean = (df_nonzero.chromatopy_ra + df_nonzero.hand_ra) / 2
old = df_nonzero.hand_ra
pos_err = 100*(df_nonzero.chromatopy_value_upper/df_nonzero.chromatopy_value)
neg_err = 100*(df_nonzero.chromatopy_value_lower/df_nonzero.chromatopy_value)

erry = (100/df_nonzero.chromatopy_ra_lower)*((df_nonzero.chromatopy_ra_upper+df_nonzero.chromatopy_ra)-(df_nonzero.chromatopy_ra-df_nonzero.chromatopy_ra_lower))/2

# Calculate bias and limits of agreement
bias = np.mean(diff)
std_diff = np.std(diff, ddof=1)
loa_upper = bias + 1.96 * std_diff
loa_lower = bias - 1.96 * std_diff

# User - User Comparison
subset = df.loc[~((df.chromatopy_ra==0) & (df.hand_ra > 0))]
subset = subset.loc[~((subset.chromatopy_ra>0) & (subset.hand_ra == 0))]
tempy = subset.fillna(0)

# Compute percent difference
tempy = tempy[~tempy['Sample Name'].isin(misidentified_samples)]
tempy['diff'] = ((tempy['chromatopy_value'] - tempy['user_2_values']) / subset['chromatopy_value']) * 100
tempy = tempy[tempy['diff']>-1000] # To fix issues related to amples with effectively 0 peak area (H1801000128 IIIc` has a peak area of 0.838 and H1801000191 GDGT-3 has a peak area of 1.444)
err = tempy['chromatopy_value_upper']+tempy['chromatopy_value_lower']
tempy['err'] = err
smaller = tempy[np.abs(tempy['diff'])<np.sqrt(err**2 + err**2)]
larger = tempy[np.abs(tempy['diff'])>np.sqrt(err**2 + err**2)]

#Check proportion of diff/error for peaks below 1000 ampl-minutes
test = tempy[tempy['chromatopy_value']<1000]
print(f"Percent of peaks below 1000 units where uncertainty > difference: {len(test[np.abs(test['diff'])<test['err']])/len(test)}")
test = tempy[tempy['chromatopy_value']>1000]
print(f"Percent of peaks above 1000 units where uncertainty > difference: {len(test[np.abs(test['diff'])<test['err']])/len(test)}")


# AX0
smaller_ax = ax0.scatter(smaller['chromatopy_value'], smaller['diff']*-1,  c='k', marker='d', ec='k', s = 70, alpha = 0.5, label = 'Difference < uncertainty')
larger_ax = ax0.scatter(larger['chromatopy_value'],larger['diff']*-1,  c='red', marker='d', ec='k', s = 70, alpha = 0.5, label = 'Difference > uncertainty')

legend_map.setdefault("User comparison\n Difference < uncertainty", smaller_ax)
legend_map.setdefault("User comparison\n Difference > uncertainty", larger_ax)
ax0.axhline(0, c = 'k')
ax0.set_ylabel('Peak Area\nDifference (%)')
ax0.legend(frameon=False)
ax0.set_title("chromatoPy user comparison")

# AX1
larger = larger[larger['chromatopy_value']<5000]
smaller = smaller[smaller['chromatopy_value']<5000]
ax1.scatter(smaller['chromatopy_value'], smaller['diff']*-1,  c='k', marker='d', ec='k', s = 70, alpha = 0.5)
ax1.scatter(larger['chromatopy_value'],larger['diff']*-1,  c='red', marker='d', ec='k', s = 70, alpha = 0.5)
ax1.set_ylabel('Peak Area\nDifference (%)')

# AX2
test = tempy[tempy['chromatopy_value']<5000]
bins = np.histogram_bin_edges(test['chromatopy_value'], bins=20)
ax2.hist([smaller['chromatopy_value'], larger['chromatopy_value']], bins=bins, color=['k', 'red'], histtype='barstacked')
ax2.set_ylabel("Peak Count")
ax2.set_xlabel('Peak Area - User 1\n(Amplitude • Time)')
df['Fig 2 D'] = df['chromatopy_value_upper']/df['chromatopy_value']

# AX3
ax3.set_title('Uncertainty')
thresh = np.std(df['chromatopy_value_upper']/df['chromatopy_value'])*2
thresh_mask_high = (df['chromatopy_value_upper']/df['chromatopy_value'])>thresh
ax3.scatter(df[thresh_mask_high]['chromatopy_value'], 
            df[thresh_mask_high]['chromatopy_value_upper']/df[thresh_mask_high]['chromatopy_value'],
            ec='k', c= 'red', alpha=0.6, zorder=2, label="Uncertainty > 2$\sigma$")
thresh_mask_low = (df['chromatopy_value_upper']/df['chromatopy_value'])<thresh
ax3.scatter(df[thresh_mask_low]['chromatopy_value'], 
            df[thresh_mask_low]['chromatopy_value_upper']/df[thresh_mask_low]['chromatopy_value'],
            ec='k', c= 'k', alpha=0.6, zorder=2, label="Uncertainty < 2$\sigma$")
ax3.axhline(np.std(df['chromatopy_value_upper']/df['chromatopy_value'])*2, c='red', linestyle='-', alpha = 0.5, zorder=1) #c='#2D728F'
ax3.set_ylabel("Relative Uncertainty (2-$\sigma$)")
ax3.legend(frameon=False)
ax4.set_xlabel("Peak Area (chromatoPy)")
ax4.set_ylabel("Peak Count")
ax3.spines[['right', 'top']].set_visible(False)
temp = df[df['chromatopy_value']<5000]
rel_uncertainty = temp['chromatopy_value_upper'] / temp['chromatopy_value']
threshold = np.std(rel_uncertainty)
below = temp[rel_uncertainty < threshold]['chromatopy_value']
above = temp[rel_uncertainty >= threshold]['chromatopy_value']
all_vals = temp['chromatopy_value']
bins = np.histogram_bin_edges(all_vals, bins=20)

# AX4
ax4.hist([below, above], bins=bins, color=['black', 'red'], histtype='barstacked')

# Draw connection lines
ax1_xmin, ax1_xmax = ax4.get_xlim()
ax1_ymin, ax1_ymax = ax4.get_ylim()
ax0_xmin, ax0_xmax = ax3.get_xlim()
ax0_ymin, ax0_ymax = ax3.get_ylim()
ax4.axvline(0, c='grey', linestyle='--', zorder=0)
ax3.axvline(0, c='grey', linestyle='--', zorder=0)
ax4.axvline(ax1_xmax, c='grey', linestyle='--', zorder=0)
ax3.axvline(ax1_xmax, c='grey', linestyle='--', zorder=0)

for x in [0, ax1_xmax]:
    con = ConnectionPatch(xyA=(x, ax1_ymax), xyB=(x, ax0_ymin), coordsA="data", coordsB="data",
                          axesA=ax4, axesB=ax3, color="grey", linestyle='--', zorder=0)
    ax4.add_artist(con)
for ax in [ax0, ax1, ax2, ax3, ax4, ax6, ax7]:
    ax.spines[['right', 'top']].set_visible(False)
    
# chromatoPy-manual comparison
temp = pd.DataFrame()
temp['mean'] = mean
temp['diff'] = diff
temp['old'] = old
temp['Region'] = df_nonzero['Region']
temp['percent'] = temp['diff']/temp.old
temp["-err"] = neg_err
temp["+err"] = pos_err
temp["erry"]=erry
temp['id'] = df_nonzero['Sample Name']
ax6.set_title("chromatoPy-manual comparison")
ax6.scatter(temp['old'], temp['percent']*100, color='k', ec = 'k', marker = 'o', alpha = 0.5, zorder=4)
ax6.axhline(0, c= 'k')
ax6.set_ylabel('Scaled Peak Area\nDifference (%)')

temp = temp[temp['old']<0.01]
ax7.scatter(temp['old'], temp['percent']*100, color='k', ec = 'k', marker = 'o', alpha = 0.5, zorder=4)
ax7.axhline(0, c= 'k')
ax7.set_ylabel('Scaled Peak Area\nDifference (%)')
ax7.set_xlabel("Scale Peak Area\n(Manual)")

ax1_xmin, ax1_xmax = ax7.get_xlim()
ax1_ymin, ax1_ymax = ax7.get_ylim()
ax0_xmin, ax0_xmax = ax6.get_xlim()
ax0_ymin, ax0_ymax = ax6.get_ylim()

for x in [0, ax1_xmax]:
    con = ConnectionPatch(xyA=(x, ax1_ymax), xyB=(x, ax0_ymin), coordsA="data", coordsB="data",
                          axesA=ax7, axesB=ax6, color="grey", linestyle='--', zorder=0)
    ax7.add_artist(con)
for ax in [ax6, ax7]:
    ax.axvline(0, c='grey', linestyle='--')
    ax.axvline(ax1_xmax, c='grey', linestyle='--')
    
# Lines over subplots
ax0.axvline(0, c='grey', linestyle='--', zorder=0)
ax0.axvline(5000, c='grey', linestyle='--', zorder=0)
    
ax1_xmin, ax1_xmax = ax1.get_xlim()
ax1_ymin, ax1_ymax = ax1.get_ylim()
ax0_xmin, ax0_xmax = ax0.get_xlim()
ax0_ymin, ax0_ymax = ax0.get_ylim()
ax2_xmin, ax2_xmax = ax2.get_xlim()
ax2_ymin, ax2_ymax = ax2.get_ylim()
xy1 = (0,ax1_ymax)
xy0 = (0,ax0_ymin)
con = ConnectionPatch(xyA=xy1, xyB=xy0, coordsA="data", coordsB="data", axesA=ax1, axesB=ax0, color="grey", linestyle='--')
ax1.add_artist(con)
xy1 = (5000,ax0_ymax)
xy0 = (5000,ax2_ymin)
con = ConnectionPatch(xyA=xy1, xyB=xy0, coordsA="data", coordsB="data",axesA=ax1, axesB=ax0, color="grey", linestyle='--')
ax1.add_artist(con)
xy1 = (5000,ax0_ymax)
xy2 = (5000,ax2_ymin)
con = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", coordsB="data",axesA=ax1, axesB=ax2, color="grey", linestyle='--')
ax2.add_artist(con)
xy1 = (0,ax1_ymax)
xy2 = (0,ax2_ymin)
con = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", coordsB="data",axesA=ax1, axesB=ax2, color="grey", linestyle='--')
ax2.add_artist(con)
ax1.set_zorder(1)
ax0.set_zorder(0)

ax5.set_axis_off()
ax8.set_axis_off()

for ax, alpha in zip([ax0, ax1, ax2, ax3, ax4, ax6, ax7], ["A", "B", "C", "D", "E", 'F', 'G']):
    ax.text(-0.1, 1.1, f"({alpha})", transform=ax.transAxes, ha='left', color='k', fontweight='bold')
    
for ax in [ax0, ax1, ax2, ax3, ax4]:
    ax.set_facecolor('none')

plt.tight_layout(rect=[0, 0, 1, 1])
fig.subplots_adjust(hspace=0.5)
plt.show()
