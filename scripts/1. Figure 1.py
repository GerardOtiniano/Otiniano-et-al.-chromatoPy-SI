"""
Python code for Figure 1.
"""
# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr


def corr_text(r, p_value):
    if p_value > 0.001:
        p_text = f'p = {p_value:.3e}'
    else:
        p_text = 'p < 0.001'
        
    if r < 0.999:
        r_text = f'r$_{{Pearson}}$ = {r:.3f}'
    else:
        r_text = 'r$_{{Pearson}}$ > 0.999'
    return r_text, p_text

# Import data
chpy_man_data = pd.read_csv('data/chromatoPy and manual data.csv') # subset of overlapping chromatoPy and manually integrated data results
chpy_full = pd.read_csv('data/chromatopy_data_full.csv') # results from chromatopy integration (no subsetting by manual integrations)
user_2_data = pd.read_csv('data/chromatoPy_user_2_data_peak_area.csv') # chromatoPy integrations from independent second user
user_2_fa = pd.read_csv('data/user_2_data_fractional_abundance.csv')

# Process data
# chpy_man_data = chpy_man_data.fillna(0)
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

                
# Figure 1
fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2)
fig.set_figheight(10)
fig.set_figwidth(10)
zeros = df.loc[(df.chromatopy_ra==0) | (df.hand_ra == 0)]
subset = df.loc[~((df.chromatopy_ra==0) & (df.hand_ra > 0))]
subset = subset.loc[~((subset.chromatopy_ra>0) & (subset.hand_ra == 0))]
legend_map = {} 

# Top left
for k in subset.Region.unique():
    temp = subset[subset.Region==k]
    temp = subset[subset.Region==k]
    y = temp['chromatopy_ra']
    low_err = temp['chromatopy_ra_lower']
    # Ensure errorbars do not extend below 0:
    low_err = np.where(y - low_err < 0, y, low_err)
    up_err = temp['chromatopy_ra_upper']
    err = [low_err, up_err]
    if k == "Lake Bolshoye Shchuchye":
        # continue
        ax1.errorbar(temp['hand_ra'], temp['chromatopy_ra'], yerr=err, c='grey', 
                     linestyle='', alpha=0.3, zorder=1)
        scatter = ax1.scatter(temp.hand_ra, temp.chromatopy_ra, color = 'grey', marker = 's', edgecolor='black', alpha=0.6, zorder=3, s=100, label = k)
        legend_map.setdefault(k, scatter)
    else:
        ax1.errorbar(temp['hand_ra'], temp['chromatopy_ra'], yerr=err, c='grey', 
                     linestyle='', alpha=0.3, zorder=1)
        scatter = ax1.scatter(temp.hand_ra, temp.chromatopy_ra, edgecolor='black', alpha=0.4, zorder=3, s=80, label = k)
        legend_map.setdefault(k, scatter)
if len(zeros) > 0:
    zero_h = ax1.scatter(zeros.hand_ra, zeros.chromatopy_ra, marker = 'x', color='k', zorder=0)
    legend_map.setdefault('Zero Values', zero_h)
lims = [
    np.min([ax1.get_xlim(), ax1.get_ylim()]),
    np.max([ax1.get_xlim(), ax1.get_ylim()])]
ax1.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

reg = LinearRegression().fit(subset.hand_ra.values.reshape(-1,1), subset.chromatopy_ra.values.reshape(-1,1))
m = reg.coef_[0][0]
b = reg.intercept_[0]
x1 = subset.hand_ra.min()
x2 = subset.hand_ra.max()
ax1.plot([x1, x2], [m*x1+b, m*x2+b], c='red', linestyle='--', zorder=1)

r, p_value = pearsonr(df.hand_ra, df.chromatopy_ra)
r_text, p_text = corr_text(r, p_value)

    
correlation_text = f'(A)\n{r_text}\n{p_text}\nn = {len(subset)}'
ax1.text(0.05, 0.75, correlation_text, transform=ax1.transAxes, ha='left', color='black')
ax1.set_xlim(-0.05, 1.05)
ax1.set_ylim(-0.05, 1.05)
ax1.set_xlabel('Scaled Peak Area\n(manual integration)')
ax1.set_ylabel('Scaled Peak Area\n(chromatoPy integration)')

# Fractional Abundance
for k in subset.Region.unique():
    temp = subset[subset.Region==k]
    temp = subset[subset.Region==k]
    y = temp['chromatopy_fa']
    low_err = temp['chromatopy_fa_lower']
    # Ensure errorbars do not extend below 0:
    low_err = np.where(y - low_err < 0, y, low_err)
    up_err = temp['chromatopy_fa_upper']
    err = [low_err, up_err]
    if k == "Lake Bolshoye Shchuchye":
        ax2.errorbar(temp['hand_fa'], temp['chromatopy_fa'], yerr=err, c='grey', 
                     linestyle='', alpha=0.3, zorder=2)
        scatter = ax2.scatter(temp.hand_fa, temp.chromatopy_fa, color = 'grey', marker = 's', edgecolor='black', alpha=0.6, zorder=3, s=100, label = k)
    else:
        ax2.errorbar(temp['hand_fa'], temp['chromatopy_fa'], yerr=err, c='grey', 
                     linestyle='', alpha=0.3, zorder=2)
        scatter = ax2.scatter(temp.hand_fa, temp.chromatopy_fa, edgecolor='black', alpha=0.4, zorder=3, s=80, label = k)
        
if len(zeros) > 0:
    ax2.scatter(zeros.hand_fa, zeros.chromatopy_fa, marker = 'x', color='k', zorder=0)

lims = [
    np.min([-1, -1]),
    np.max([2, 2])]
ax2.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

reg = LinearRegression().fit(subset.hand_fa.values.reshape(-1,1), subset.chromatopy_fa.values.reshape(-1,1))
m = reg.coef_[0][0]
b = reg.intercept_[0]
x1 = subset.hand_fa.min()
x2 = subset.hand_fa.max()
ax2.plot([x1, x2], [m*x1+b, m*x2+b], c='red', linestyle='--', zorder=1)
ax2.set_xlim(-0.05, 1.05)
ax2.set_ylim(-0.05, 1.05)

r, p_value = pearsonr(df.hand_fa, df.chromatopy_fa)
r_text, p_text = corr_text(r, p_value)
    
correlation_text = f'(B)\n{r_text}\n{p_text}\nn = {len(subset)}'
ax2.text(0.05, 0.75, correlation_text, transform=ax2.transAxes, ha='left', color='black')
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax2.set_xlabel('Fractional Abundance\n(manual integration)')

ax2.set_ylabel('Fractional Abundance\n(chromatoPy integration)', rotation=270, labelpad=25)
# ax3
comparison = ax3.scatter(subset.chromatopy_value_PA, subset.user_2_values, c='k', marker='d', label = "Inter-user comparison", s=80)
legend_map["Inter-user comparison"] = comparison
r, p_value = pearsonr(subset.chromatopy_value_PA, subset.user_2_values)
r_text, p_text = corr_text(r, p_value)

correlation_text = f'(C)\n{r_text}\n{p_text}\nn = {len(subset)}'
ax3.text(0.05, 0.75, correlation_text, transform=ax3.transAxes, ha='left', color='black')
reg = LinearRegression().fit(subset.chromatopy_value_PA.values.reshape(-1,1),subset.user_2_values.values.reshape(-1,1))
m = reg.coef_[0][0]
b = reg.intercept_[0]
x1 = subset.chromatopy_value_PA.min()
x2 = subset.chromatopy_value_PA.max()
regression_line, = ax3.plot([x1, x2], [m*x1+b, m*x2+b], c='red', linestyle='--', zorder=1, label="Regression model")
legend_map["Regression line"] = regression_line

lims = [
    np.min([-10000, -10000]),
    np.max([700000, 700000]),
]
one_to_one, = ax3.plot(lims, lims, 'k-', alpha=0.75, zorder=0, label="1:1")
legend_map["1:1 line"] = one_to_one
ax3.set_xlim(-10000, 500000)
ax3.set_ylim(-10000, 500000)
ax3.set_xlabel("Peak Area\n(chromatoPy, user 1)")
ax3.set_ylabel("Peak Area\n(chromatoPy, user 2)")

df=user_2_fa.fillna(0)
raw = df
sample_to_ignore = ['H2202081', 'H2308012', 'H2202085', 'H2202087', 'H1608000014', 'H1608000013', 'H2307071']
ignored = df[df['Sample Name'].isin(sample_to_ignore)]
df = df[~df['Sample Name'].isin(sample_to_ignore)]
comparison = ax4.scatter(df.chromatopy_fa, df.user_2_fa, c='k', marker='d', label = "Inter-user comparison", s=80)
comparison_ignore = ax4.scatter(ignored.chromatopy_fa, ignored.user_2_fa, c='red', marker='x', label = "Discordant GDGTs", s=40)
legend_map["Inter-user comparison"] = comparison
legend_map["Discordant GDGT peaks"] = comparison_ignore

r, p_value = pearsonr(raw.chromatopy_fa, raw.user_2_fa)
r_text, p_text = corr_text(r, p_value)

correlation_text = f'(D)\n{r_text}\n{p_text}\nn = {len(raw)}'
ax4.text(0.05, 0.75, correlation_text, transform=ax4.transAxes, ha='left', color='black')
reg = LinearRegression().fit(df.chromatopy_fa.values.reshape(-1,1),df.user_2_fa.values.reshape(-1,1))
m = reg.coef_[0][0]
b = reg.intercept_[0]
x1 = df.chromatopy_fa.min()
x2 = df.chromatopy_fa.max()
regression_line, = ax4.plot([x1, x2], [m*x1+b, m*x2+b], c='red', linestyle='--', zorder=1, label="Regression model")
legend_map["Regression line"] = regression_line
lims = [np.min([0, 0]),np.max([1, 1]),]
one_to_one, = ax4.plot(lims, lims, 'k-', alpha=0.75, zorder=0, label="1:1")
legend_map["1:1 line"] = one_to_one
ax4.set_xlim(-0.05, 1.05)
ax4.set_ylim(-0.05, 1.05)
ax4.set_xlabel("Fractional Abundance\n(chromatoPy, user 1)")
ax4.set_ylabel("Fractional Abundance\n(chromatoPy, user 2)")
ax4.yaxis.set_label_position("right")
ax4.yaxis.tick_right()
  
plt.tight_layout(rect=[0, 0.15, 1, 1])
fig.legend(handles = list(legend_map.values()), labels = list(legend_map.keys()), loc = 'center',
    bbox_to_anchor=(0.5, 0.1), ncol = 3, frameon = False)    
plt.savefig('/Users/gerard/Documents/GitHub/chromatoPy_manuscript/Figures/mod/Fig 1.png', dpi=300)
plt.show()