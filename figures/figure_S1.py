import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde
from data_processing.load_data import load_chraomtopy_and_manual

# Import data
df  = load_chraomtopy_and_manual()
x_v = 'chromatopy_pa'
y_v = 'user_2_pa'
brgdgts = ["Ia", "IIa", "IIa'", "IIIa", "IIIa'", "Ib", "IIb", "IIb'", "IIIb", "IIIb'", "Ic", "IIc", "IIc'", "IIIc", "IIIc'"]
sample_to_ignore = ['H2202081', 'H2308012', 'H2202085', 'H2202087', 'H1608000014', 'H1608000013', 'H2307071'] # misidentified peaks
ignored_samples = df.loc[((df['Sample Name'].isin(sample_to_ignore))&(df['variable'].isin(brgdgts)))]
df = df[~df['Sample Name'].isin(sample_to_ignore)]
df = df.loc[~((df['Sample Name'].isin(sample_to_ignore))&(df['variable'].isin(brgdgts)))]

all_regions = sorted(set(df['Region'].dropna().unique()))
color_list = plt.cm.get_cmap('tab20', len(all_regions)).colors
region_color_map = dict(zip(all_regions, color_list))
brgdgt_types = ["Ia", "IIa","IIa'" , "IIIa", "IIIa'",
                "Ib", "IIb", "IIb'", "IIIb", "IIIb'",
                "Ic", "IIc", "IIc'", "IIIc", "IIIc'", 
                'GDGT-0', 'GDGT-1', 'GDGT-2', 'GDGT-3', 'GDGT-4', "GDGT-4'"]
unique_types = sorted(df['variable'].unique())

plt.rcParams.update({'font.size': 10})
fig, axes = plt.subplots(4, 6, figsize=(15, 11), constrained_layout=True)

# legend
unique_handles = []
unique_labels = []
seen_labels = set()

# Iterate over GDGT types and axes
x=0
for i in range(4):
    for j in range(6):
        if j == 5 and i != 3:
            axes[i, j].axis("off")
            continue
        ax = axes[i, j]
        if x >= len(brgdgt_types):
            ax.axis("off")
            continue
        brgdgt_type = brgdgt_types[x]
        ax = axes[i,j]
        subset = df[df['variable'] == brgdgt_type]
        ig_sub = ignored_samples[ignored_samples['variable']== brgdgt_type]
        # limits
        z_min_p = subset[[x_v,y_v]].min().min()

        z_max_p = subset[[x_v,y_v]].max().max()
        fudge = (z_max_p-z_min_p)*0.05
        z_min = z_min_p-fudge
        z_max = z_max_p+fudge
        
        zeros = subset.loc[(subset[x_v]==0) & (subset[y_v] > 0)]
        zeros_man = subset.loc[(subset[x_v]>0) & (subset[y_v] == 0)]
        subset = subset.loc[~(subset[x_v]==0) & (subset[y_v] > 0)]
        for k in subset.Region.unique():
            temp = subset[subset.Region==k]
            y = temp[y_v]
            scatter = ax.scatter(temp[x_v], temp[y_v], color='k', edgecolor='black', marker='d', alpha=0.6, zorder=3, s=50, label = k)
            if "GDGT peak" not in seen_labels:
                unique_handles.append(scatter)
                unique_labels.append("GDGT peak")
                seen_labels.add("GDGT peak")
            ig_scatter = ax.scatter(ig_sub[x_v], ig_sub[y_v], color = 'red', marker='x')
            if 'Discordant GDGT peaks' not in seen_labels:
                unique_handles.append(ig_scatter)
                unique_labels.append('Discordant GDGT peaks')
                seen_labels.add('Discordant GDGT peaks')
        if len(zeros) > 0:
            zero_h = ax.scatter(zeros[y_v], zeros[x_v], marker = 'x', color='k', zorder=2)
            if 'Discordant GDGT peaks' not in seen_labels:
                unique_handles.append(zero_h)
                unique_labels.append('Zero values')
                seen_labels.add('Zero values')
        if len(zeros_man) > 0:
            # print(f"{brgdgt_type}: {len(zeros_man)}")
            ax.scatter(zeros_man[y_v], zeros_man[x_v], marker = 'x', color='k', zorder=2)
        one_to_one = ax.plot([z_min_p,z_max_p], [z_min_p,z_max_p], 'k', alpha=0.75, zorder=0)[0]
        if "1:1 line" not in seen_labels:
            unique_handles.append(one_to_one)
            unique_labels.append("1:1 line")
            seen_labels.add("1:1 line")
        
        # Correlation
        r, p_value = pearsonr(subset[x_v], subset[y_v])
        r  = r
        brgdgt_type = brgdgt_type.replace("'", "$^{\prime}$")
        p_text = f'p = {p_value:.3f}'
        r_text = f'$r_{{Pearson}}$ = {r:.3f}'
        non_zero_data = subset.loc[~(subset[y_v]==0) & (subset[x_v]> 0)]
        reg = LinearRegression().fit(non_zero_data[y_v].values.reshape(-1,1), non_zero_data[x_v].values.reshape(-1,1))
        m = reg.coef_[0][0]
        b = reg.intercept_[0]
        x1 = non_zero_data[x_v].min()
        x2 = non_zero_data[x_v].max()
        reg_line = ax.plot([x1, x2], [m*x1+b, m*x2+b], c='red', linestyle='--', zorder=1)[0]
        if 'Regression line' not in seen_labels:
            unique_handles.append(reg_line)
            unique_labels.append('Regression line')
            seen_labels.add('Regression line')
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
all_handles = unique_handles 
all_labels = unique_labels
fig.legend(
    handles = all_handles, 
    labels  = all_labels, 
    loc     = 'lower center',
    bbox_to_anchor=(0.5, 0),
    ncol    = 4,
    frameon = False)
fig.supxlabel('Peak Area\n(User 1)', x = 0.54, y = 0.07, ha = 'center')
fig.supylabel('Peak Area\n(User 2)', x=0.01)
plt.tight_layout(rect=[0, 0.06, 1, 0.95])
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig("Fig. S1.png", dpi=300)
plt.show()
