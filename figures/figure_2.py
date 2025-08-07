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

legend_map = {} 
plt.rcParams.update({'font.size': 10})

# Import data
df, _ = load_chraomtopy_and_manual()
# Calcualte error, bias, and limits of agreement between manual and 
results = bland_altman_metrics(df)
# Calcaulte difference and error metrics of user comparison
smaller, larger = user_comparison(df)


fig, axs = plt.subplots(3, 3, figsize=(15, 7), constrained_layout=True)
ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axs[0,0], axs[1,0], axs[2,0], axs[0,1], axs[1,1], axs[2,1], axs[0,2], axs[1,2], axs[2,2]

# Panel A
smaller_ax = ax0.scatter(smaller['chromatopy_pa'], -smaller['diff'], c='k', marker='d', 
                         ec='k', s=70, alpha=0.5, label='Difference < uncertainty')
larger_ax = ax0.scatter(larger['chromatopy_pa'], -larger['diff'], c='red', marker='d',
                        ec='k', s=70, alpha=0.5, label='Difference > uncertainty')
ax0.axhline(0, c='k')
ax0.set_ylabel('Peak Area\nDifference (%)')
ax0.set_title("chromatoPy user comparison")
ax0.legend(frameon=False)

# Panel B
filt_sm = smaller[smaller['chromatopy_pa'] < 5000]
filt_lg = larger[larger['chromatopy_pa'] < 5000]
ax1.scatter(filt_sm['chromatopy_pa'], -filt_sm['diff'], c='k', marker='d', ec='k', s=70, alpha=0.5)
ax1.scatter(filt_lg['chromatopy_pa'], -filt_lg['diff'], c='red', marker='d', ec='k', s=70, alpha=0.5)
ax1.set_ylabel('Peak Area\nDifference (%)')

# Panel C
test = df[df['chromatopy_pa']<5000]
bins = np.histogram_bin_edges(test['chromatopy_pa'], bins=20)
ax2.hist([filt_sm['chromatopy_pa'], filt_lg['chromatopy_pa']], bins=bins, color=['k', 'red'], histtype='barstacked')
ax2.set_ylabel("Peak Count")
ax2.set_xlabel('Peak Area - User 1\n(Amplitude • Time)')

# Panel D
below, above, thresh = threshold_outliers(results) # get data bove and below outlier threshold (2-sigma)
ax3.set_title('Uncertainty')
ax3.scatter(below['chromatopy_pa'], results['pos_err'][below.index]/100, c='k', ec='k', alpha=0.6, label="Uncertainty < 2$σ$")
ax3.scatter(above['chromatopy_pa'], results['pos_err'][above.index]/100, c='red', ec='k', alpha=0.6, label="Uncertainty > 2$σ$")
ax3.axhline(thresh / 100, c='red', linestyle='-', alpha=0.5)
ax3.set_ylabel("Relative Uncertainty (2-$σ$)")
ax3.legend(frameon=False)

# Panel E
ax4.hist([below['chromatopy_pa'], above['chromatopy_pa']], bins=bins, color=['black', 'red'], histtype='barstacked')
ax4.set_xlabel("Peak Area (chromatoPy)")
ax4.set_ylabel("Peak Count")

# Panel F
ax6.set_title("chromatoPy-manual comparison")
ax6.scatter(results['mean'], results['diff'] / results['clean_df']['hand_ra'] * 100, color='k', ec='k', marker='o', alpha=0.5)
ax6.axhline(0, c='k')
ax6.set_ylabel("Scaled Peak Area\nDifference (%)")

# Panel G
small = results['clean_df'][results['clean_df']['hand_ra'] < 0.01]
ax7.scatter(small['hand_ra'], (small['chromatopy_ra'] - small['hand_ra']) / small['hand_ra'] * 100, color='k', ec='k', marker='o', alpha=0.5)
ax7.axhline(0, c='k')
ax7.set_ylabel("Scaled Peak Area\nDifference (%)")
ax7.set_xlabel("Scale Peak Area\n(Manual)")

# Clean spines
for ax in [ax0, ax1, ax2, ax3, ax4, ax6, ax7]:
    clean_spines(ax)

# Annotate subplots
for ax, alpha in zip([ax0, ax1, ax2, ax3, ax4, ax6, ax7], ["A", "B", "C", "D", "E", 'F', 'G']):
    ax.text(-0.1, 1.1, f"({alpha})", transform=ax.transAxes, ha='left', color='k', fontweight='bold')

# Connection Lines - Panels A-C
ax0.axvline(0, c='grey', linestyle='--', zorder=0)
ax0.axvline(5000, c='grey', linestyle='--', zorder=0)
ax0_ymin, ax0_ymax = ax0.get_ylim()
ax1_ymin, ax1_ymax = ax1.get_ylim()
ax2_ymin, ax2_ymax = ax2.get_ylim()
connect_vertical_patch(ax1, ax0, 0, ax1_ymax, ax0_ymin)
connect_vertical_patch(ax1, ax0, 5000, ax0_ymax, ax2_ymin)
connect_vertical_patch(ax1, ax2, 5000, ax0_ymax, ax2_ymin)
connect_vertical_patch(ax1, ax2, 0, ax1_ymax, ax2_ymin)

# Connection Lines - Panels D, E
plot_connections([ax3,ax4])

# Connection Lines - Panels F, G
plot_connections([ax6,ax7])
    
# Aesthetics 
for ax, alpha in zip([ax0, ax1, ax2, ax3, ax4, ax6, ax7], ["A", "B", "C", "D", "E", 'F', 'G']):
    ax.text(-0.1, 1.1, f"({alpha})", transform=ax.transAxes, ha='left', color='k', fontweight='bold')
    
for ax in [ax0, ax1, ax2, ax3, ax4, ax5, ax6,ax7]:
    ax.set_facecolor('none')

# Hide unused axes
ax5.set_axis_off()
ax8.set_axis_off()

plt.tight_layout(rect=[0, 0, 1, 1])
fig.subplots_adjust(hspace=0.5)
plt.show()