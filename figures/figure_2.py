"""
Python code for Figure 2.
"""
# Import packages
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.patches import ConnectionPatch
import numpy as np
from data_processing.load_data import *
from figures.figure_2_functions import *

plt.rcParams.update({'font.size': 10})

# Import data
df = load_chraomtopy_and_manual()

# Remove misidentified samples
ignore_isogdgts = ['H1608000189']
ignore_brgdgts = ['H1608000014', 'H2202085', 'H2202081', 'H2202087', 'H1608000013', 'H1805000004', 'H2307064', 'H2204051']
df, ignored = remove_samples(df, ignore_isogdgts, ignore_brgdgts)

# Calcaulte difference and error metrics of user comparison
df = user_comparison(df)
df_low = df[df['diff']/df['err']<1]
df_high = df[df['diff']/df['err']>1]
df["binary_diff"] = np.where(df["diff"] / df["err"] < 1, 0, 1)

fig, axs = plt.subplots(3, 3, figsize=(15, 7), constrained_layout=True)
ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axs[0,0], axs[1,0], axs[2,0], axs[0,1], axs[1,1], axs[2,1], axs[0,2], axs[1,2], axs[2,2]

# Panel A
ax0.scatter(df_low['chromatopy_pa'], df_low['perc_diff'], c='k', marker='d', ec='k', 
            s=70, alpha=0.5, label='Difference < uncertainty')
ax0.scatter(df_high['chromatopy_pa'], df_high['perc_diff'], c='red', marker='d', ec='k',
            s=70, alpha=0.5, label='Difference > uncertainty')
ax0.axhline(0, c='k')
ax0.set_ylabel('Peak Area\nDifference (%)')
ax0.set_title("chromatoPy user comparison")

# Panel B
sub = df[df['chromatopy_pa']<5000]
ax1.scatter(sub[sub['binary_diff']==0]['chromatopy_pa'], sub[sub['binary_diff']==0]['perc_diff'], c='k', marker='d', ec='k', s=70, alpha=0.5)
ax1.scatter(sub[sub['binary_diff']==1]['chromatopy_pa'], sub[sub['binary_diff']==1]['perc_diff'], c='red', marker='d', ec='k', s=70, alpha=0.5)
ax1.set_ylabel('Peak Area\nDifference (%)')

# Panel C
bins = np.histogram_bin_edges(sub['chromatopy_pa'], bins=20)
ax2.hist([sub[sub['binary_diff']==0]['chromatopy_pa'], sub[sub['binary_diff']==1]['chromatopy_pa']], bins=bins, color=['k','red'], histtype='barstacked')
ax2.set_ylabel("Peak Count")
ax2.set_xlabel('Peak Area - User 1\n(Amplitude • Time)')

# Panel D
ax3.set_title('Uncertainty')
ax3.scatter(df_low['chromatopy_pa'], df_low['rel_err'], c='k', marker='d', ec='k', s=70, alpha=0.5)
ax3.scatter(df_high['chromatopy_pa'], df_high['rel_err'], c='red', marker='d', ec='k', s=70, alpha=0.5)
ax3.set_ylabel("Relative Uncertainty\n(%)")

# # Panel E
sub = df[df['chromatopy_pa']<5000]
ax4.scatter(sub[sub['binary_diff']==0]['chromatopy_pa'], sub[sub['binary_diff']==0]['rel_err'], c='k', marker='d', ec='k', s=70, alpha=0.5)
ax4.scatter(sub[sub['binary_diff']==1]['chromatopy_pa'], sub[sub['binary_diff']==1]['rel_err'], c='red', marker='d', ec='k', s=70, alpha=0.5)
ax4.set_xlabel("Peak Area (chromatoPy)")
ax4.set_ylabel("Relative Uncertainty\n(%)")

# Panel F
edges = np.arange(0, 130 + 5, 5)  # 0,15,...,150
labels = [f"{int(a)}–{int(b)}%" for a, b in zip(edges[:-1], edges[1:])]
ypos   = np.arange(len(labels))
rel = pd.to_numeric(df["rel_err"], errors="coerce").abs()
flag = pd.to_numeric(df["binary_diff"], errors="coerce").astype("Int64")
mask = rel.notna() & flag.notna()
rel  = rel[mask]
flag = flag[mask].astype(int)
rel_in  = rel[flag == 0]
rel_out = rel[flag == 1]
cnt_in,  _ = np.histogram(rel_in,  bins=edges)
cnt_out, _ = np.histogram(rel_out, bins=edges)
keep = (cnt_in + cnt_out) > 0
cnt_in, cnt_out = cnt_in[keep], cnt_out[keep]
labels = np.array(labels)[keep]
ypos   = np.arange(len(labels))
ax5.barh(ypos, cnt_in,  color="k",   label="Difference < uncertainty")
ax5.barh(ypos, cnt_out, left=cnt_in, color="red", label="Difference > uncertainty")
# ax5.set_xscale("log")
# ax5.set_xlim(10e-2, None)
ax5.set_yticks(ypos, labels)
ax5.invert_yaxis()
ax5.set_xlabel("Peak count")
ax5.set_ylabel("Relative Uncertainty (%)")

# Panel G
ax6.set_title("chromatoPy-manual comparison")
clean_df = df
clean_df['user-chromatopy'] = ((clean_df['chromatopy_ra']-clean_df['hand_ra'])/((clean_df['chromatopy_ra']+clean_df['hand_ra'])/2))*100
ax6.scatter(clean_df['chromatopy_ra'], clean_df['user-chromatopy'], c='k', 
            ec='k', alpha=0.5, s=70, label="chromatoPy-manual comparison")
ax6.axhline(0, c='k')
ax6.set_ylabel("Scaled Peak Area\nDifference (%)")

# Panel H
small = clean_df[clean_df['hand_ra']<0.01]
ax7.scatter(small['chromatopy_ra'], small['user-chromatopy'], c='k', ec='k', alpha=0.5, s=70)
ax7.axhline(0, c='k')
ax7.set_ylabel("Scaled Peak Area\nDifference (%)")
ax7.set_xlabel("Scale Peak Area\n(Manual)")

# Clean spines
for ax in [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
    clean_spines(ax)


# Connection Lines - Panels A-C
ax0.axvline(0, c='grey', linestyle='--', zorder=0)
ax0.axvline(5000, c='grey', linestyle='--', zorder=0)
ax0_ymin, ax0_ymax = ax0.get_ylim()
ax1_ymin, ax1_ymax = ax1.get_ylim()
ax2_ymin, ax2_ymax = ax2.get_ylim()
connect_vertical_patch(ax1, ax0, 0, ax1_ymax, ax0_ymin)
connect_vertical_patch(ax1, ax0, 5000, ax0_ymax, ax1_ymin)
connect_vertical_patch(ax1, ax2, 5000, ax0_ymax, ax2_ymin)
connect_vertical_patch(ax1, ax2, 0, ax1_ymax, ax2_ymin)

# Connection Lines - Panels D, E
plot_connections([ax3,ax4])

# Connection Lines - Panels F, G
plot_connections([ax6,ax7])
    
# Aesthetics 
for ax, alpha in zip([ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7], ["A", "B", "C", "D", "E", 'F', 'G', 'H']):
    ax.text(-0.1, 1.1, f"({alpha})", transform=ax.transAxes, ha='left', color='k', fontweight='bold')
    
for ax in [ax0, ax1, ax2, ax3, ax4, ax5, ax6,ax7]:
    ax.set_facecolor('none')

# Hide unused axes
ax8.set_axis_off()

# Legend
handles0, labels0 = ax0.get_legend_handles_labels()
handles6, labels6 = ax6.get_legend_handles_labels()
handles = handles0 + handles6
labels = labels0 + labels6
ax8.legend(handles, labels, frameon=False, loc="center")
ax8.axis("off")

plt.tight_layout(rect=[0, 0, 1, 1])
fig.subplots_adjust(hspace=0.5)
plt.savefig("Fig. 2.png", dpi=300)
plt.show()