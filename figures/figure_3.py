"""
Python code for Figure 3.
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


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


brgdgt_types = ["Ia", "IIa","IIa'" , "IIIa", "IIIa'",
                "Ib", "IIb", "IIb'", "IIIb", "IIIb'",
                "Ic", "IIc", "IIc'", "IIIc", "IIIc'", 
                'GDGT-0', 'GDGT-1', 'GDGT-2', 'GDGT-3', 'GDGT-4', "GDGT-4'"]
unique_types = sorted(df['variable'].unique())

plt.rcParams.update({'font.size': 10})
# Set up the 4x4 subplot grid with space for 16 subplots (bottom-right subplot blank)
fig, axes = plt.subplots(4, 6, figsize=(15, 11), constrained_layout=True)

# legend
unique_handles = []
unique_labels = []


legend_map = {} 
# Iterate over GDGT types and axes
x=0
for i in range(4):
    for j in range(6):
        if j == 5 and i != 3:
            axes[i, j].axis("off")
            continue
        ax = axes[i, j]
        if x >= len(brgdgt_types):  # stop if we've used all types
            ax.axis("off")
            continue
        brgdgt_type = brgdgt_types[x]
        ax = axes[i,j]
        subset = df[df['variable'] == brgdgt_type].copy()
        z_min_p = subset[['chromatopy_ra','hand_ra']].min().min()
        subset.loc[:, 'upper_limit'] = subset['chromatopy_ra'] + subset['chromatopy_ra_upper']
        z_max_p = subset[['hand_ra','upper_limit']].max().max()#subset[['chromatopy_ra','hand_ra']].max().max()
        fudge = (z_max_p-z_min_p)*0.05
        z_min = z_min_p-fudge
        z_max = z_max_p+fudge
        
        zeros = subset.loc[(subset.chromatopy_ra==0) & (subset.hand_ra > 0)]
        zeros_man = subset.loc[(subset.chromatopy_ra>0) & (subset.hand_ra == 0)]
        subset = subset.loc[~(subset.chromatopy_ra==0) & (subset.hand_ra > 0)]
        for k in subset.Region.unique():
            temp = subset[subset.Region==k]
            y = temp['chromatopy_ra']
            low_err = temp['chromatopy_ra_lower']
            up_err = temp['chromatopy_ra_upper']
            low_err = np.where(y - low_err < 0, y, low_err)
            up_err = np.where(y + up_err >= 1, y, up_err)
            err = [low_err, up_err]
            if k == "Lake Bolshoye Shchuchye":
                ax.errorbar(temp['hand_ra'], temp['chromatopy_ra'], yerr=err, c='grey', 
                             linestyle='', alpha=0.3, zorder=1)
                scatter = ax.scatter(temp.hand_ra, temp.chromatopy_ra, color = 'grey', marker = 's', edgecolor='black', alpha=0.75, zorder=3, s=50, label = k)
                legend_map.setdefault(k, scatter)
            
            else:
                ax.errorbar(temp['hand_ra'], temp['chromatopy_ra'], yerr=err, c='grey', 
                             linestyle='', alpha=0.3, zorder=1)
                scatter = ax.scatter(temp.hand_ra, temp.chromatopy_ra, edgecolor='black', alpha=0.6, zorder=3, s=50, label = k)
                legend_map.setdefault(k, scatter)
            if k not in unique_labels:
                unique_handles.append(scatter)
                unique_labels.append(k)
        if len(zeros) > 0:
            print(f"{brgdgt_type}: {len(zeros)}")
            zero_h = ax.scatter(zeros.hand_ra, zeros.chromatopy_ra, marker = 'x', color='k', zorder=2)
            legend_map.setdefault('Zero Values', zero_h)

        if len(zeros_man) > 0:
            print(f"{brgdgt_type}: {len(zeros_man)}")
            ax.scatter(zeros_man.hand_ra, zeros_man.chromatopy_ra, marker = 'x', color='k', zorder=2)

        one_one_line = ax.plot([z_min_p,z_max_p], [z_min_p,z_max_p], 'k', alpha=0.75, zorder=0, label = "1:1 line")
        legend_map["1:1 line"] = one_one_line[0]
        # Correlation
        r, p_value = pearsonr(subset.hand_ra, subset.chromatopy_ra)
        r  = r
        brgdgt_type = brgdgt_type.replace("'", r"$^{\prime}$")
        p_text = f'p = {p_value:.3f}'
        r_text = f'$r_{{Pearson}}$ = {r:.3f}'
        
        non_zero_data = subset.loc[~(subset.chromatopy_ra==0) & (subset.hand_ra > 0)]
        reg = LinearRegression().fit(non_zero_data.hand_ra.values.reshape(-1,1), non_zero_data.chromatopy_ra.values.reshape(-1,1))
        m = reg.coef_[0][0]
        b = reg.intercept_[0]
        x1 = non_zero_data.hand_ra.min()
        x2 = non_zero_data.hand_ra.max()
        regression_line = ax.plot([x1, x2], [m*x1+b, m*x2+b], c='red', linestyle='--', zorder=1, label='Regression line')
        legend_map["Regression line"] = regression_line[0]
        if p_value < 0.001:
            p_text = 'p < 0.001'
        if r > 0.999:
            r_text = '$r_{{Pearson}}$ > 0.999'
        correlation_text = f'{brgdgt_type}\n{r_text}\n{p_text}\nn = {len(subset)}'
        
        x_min = ax.get_xbound()[0]
        y_max = ax.get_ybound()[1]
        y_min = ax.get_ybound()[0]
        ax.text(0,1.01, brgdgt_type, transform=ax.transAxes)
        correlation_text = f'{r_text}\n{p_text}\nn = {len(subset)}'
        ax.text(0.05, 0.65, correlation_text, transform=ax.transAxes, ha='left', fontsize=10, color='black')
        ax.set_xlim(z_min, z_max)
        ax.set_ylim(z_min, z_max)
        x += 1
        
# Legend
fig.legend(
    handles = list(legend_map.values()),
    labels  = list(legend_map.keys()),
    loc     = 'lower center',
    bbox_to_anchor=(0.5, 0),
    ncol    = 3,
    frameon = False)
fig.supxlabel('Scaled Peak Area\n(Manual)', x = 0.54, y = 0.11, ha = 'center')
fig.supylabel('Scaled Peak Area\n (chromatoPy)', x=0.01)

plt.tight_layout(rect=[0, 0.1, 1, 0.95])
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig('/Users/gerard/Documents/GitHub/chromatoPy_manuscript/Figures/mod/Fig 3.png', dpi=300)
plt.show()