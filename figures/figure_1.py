import matplotlib.pyplot as plt
from data_processing.load_data import *
from figures.figure_1_functions import *


# Import data
df = load_chraomtopy_and_manual() # imported data and dataframe containing misidentified samples between users
zeros = df.loc[(df.chromatopy_ra==0) | (df.hand_ra == 0)] # Undected peaks by chraomtopy and manual
subset = df.loc[~((df.chromatopy_ra==0) & (df.hand_ra > 0))] # Remove peaks detected manually but not by chromatopy
subset = subset.loc[~((subset.chromatopy_ra>0) & (subset.hand_ra == 0))] # Remove peaks detected by chromatopy but not manually

fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(10, 10))
legend_map = {}

# AX A
legend_map.update(plot_scatter_with_errorbars(ax1, subset, 'hand_ra', 'chromatopy_ra',
                                               'chromatopy_ra_lower', 'chromatopy_ra_upper'))
if len(zeros) > 0:
    zero_h = plot_zero_marker(ax1, zeros, 'hand_ra', 'chromatopy_ra')
    legend_map.setdefault('Zero Values', zero_h)
add_one_to_one_line(ax1, [0, 1])
add_regression_line(ax1, subset.hand_ra, subset.chromatopy_ra)
add_correlation_text(ax1, subset.hand_ra, subset.chromatopy_ra, subset,'A')
ax1.set_xlim(-0.05, 1.05)
ax1.set_ylim(-0.05, 1.05)
ax1.set_xlabel('Scaled Peak Area\n(manual integration)')
ax1.set_ylabel('Scaled Peak Area\n(chromatoPy integration)')

# AX B
legend_map.update(plot_scatter_with_errorbars(ax2, subset, 'hand_fa', 'chromatopy_fa',
                                               'chromatopy_fa_lower', 'chromatopy_fa_upper'))
if len(zeros) > 0:
    plot_zero_marker(ax2, zeros, 'hand_fa', 'chromatopy_fa')

add_one_to_one_line(ax2, [0, 1])
add_regression_line(ax2, subset.hand_fa, subset.chromatopy_fa)
add_correlation_text(ax2, subset.hand_fa, subset.chromatopy_fa, subset,'B')
ax2.set_xlim(-0.05, 1.05)
ax2.set_ylim(-0.05, 1.05)
ax2.set_xlabel('Fractional Abundance\n(manual integration)')
ax2.set_ylabel('Fractional Abundance\n(chromatoPy integration)', rotation=270, labelpad=25)
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()

# AX C
ignore_isogdgts = ['H1608000189']
ignore_brgdgts = ['H1608000014', 'H2202085', 'H2202081', 'H2202087', 'H1608000013', 'H1805000004', 'H2307064', 'H2204051']
subset_user, ignored = remove_samples(subset, ignore_isogdgts=ignore_isogdgts, ignore_brgdgts=ignore_brgdgts)

comparison = ax3.scatter(subset_user.chromatopy_pa, subset_user.user_2_pa, c='k', marker='d', s=80, label='Inter-user comparison')
comparison_ignore = ax3.scatter(ignored.chromatopy_pa, ignored.user_2_pa, c='red', marker='x', s=40)
legend_map["Inter-user comparison"] = comparison
add_regression_line(ax3, subset.chromatopy_pa, subset.user_2_pa)
add_correlation_text(ax3, subset.chromatopy_pa, subset.user_2_pa, subset,'C')
lims = [-10000, 500000]
add_one_to_one_line(ax3, lims)
ax3.set_xlim(lims)
ax3.set_ylim(lims)
ax3.set_xlabel("Peak Area\n(chromatoPy, user 1)")
ax3.set_ylabel("Peak Area\n(chromatoPy, user 2)")

# AX D 
comparison = ax4.scatter(subset_user.chromatopy_fa, subset_user.user_2_fa, c='k', marker='d', s=80, label='Inter-user comparison')
comparison_ignore = ax4.scatter(ignored.chromatopy_fa, ignored.user_2_fa, c='red', marker='x', s=40, label='Discordant GDGTs')
legend_map["Inter-user comparison"] = comparison
legend_map["Discordant GDGT peaks"] = comparison_ignore

add_regression_line(ax4, subset.chromatopy_fa, subset.user_2_fa)
add_correlation_text(ax4, subset.chromatopy_fa, subset.user_2_fa, subset,'D')
add_one_to_one_line(ax4, [0, 1])
ax4.set_xlim(-0.05, 1.05)
ax4.set_ylim(-0.05, 1.05)
ax4.set_xlabel("Fractional Abundance\n(chromatoPy, user 1)")
ax4.set_ylabel("Fractional Abundance\n(chromatoPy, user 2)")
ax4.yaxis.set_label_position("right")
ax4.yaxis.tick_right()

plt.tight_layout(rect=[0, 0.15, 1, 1])
fig.legend(handles=list(legend_map.values()), labels=list(legend_map.keys()),
           loc='center', bbox_to_anchor=(0.5, 0.1), ncol=3, frameon=False)
plt.savefig("Fig. 1.png", dpi=300)
plt.show()