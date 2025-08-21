import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

def plot_scatter_with_errorbars(ax, df, x_col, y_col, yerr_lower_col, yerr_upper_col,
                                region_col='Region', special_region='Lake Bolshoye Shchuchye',
                                marker='o', legend_map=None):
    if legend_map is None:
        legend_map = {}
    for region in df[region_col].unique():
        temp = df[df[region_col] == region]
        y = temp[y_col]
        low_err = temp[yerr_lower_col]
        up_err = temp[yerr_upper_col]
        err = [low_err, up_err]

        ax.errorbar(temp[x_col], temp[y_col], yerr=err, c='grey', linestyle='', alpha=0.3, zorder=1)
        if region == special_region:
            scatter = ax.scatter(temp[x_col], temp[y_col], color='grey', marker='s', edgecolor='black',
                                 alpha=0.6, zorder=3, s=100, label=region)
        else:
            scatter = ax.scatter(temp[x_col], temp[y_col], edgecolor='black', alpha=0.4, zorder=3, s=80,
                                 label=region)
        legend_map.setdefault(region, scatter)
    return legend_map

def add_regression_line(ax, x, y, color='red', label='Regression model', linestyle='--', zorder=1):
    reg = LinearRegression().fit(x.values.reshape(-1, 1), y.values.reshape(-1, 1))
    m = reg.coef_[0][0]
    b = reg.intercept_[0]
    x1, x2 = x.min(), x.max()
    return ax.plot([x1, x2], [m * x1 + b, m * x2 + b], color=color, linestyle=linestyle,
                   zorder=zorder, label=label)[0]

def add_correlation_text(ax, x, y, subset, panel_label, pos=(0.05, 0.75), color='black'):
    r, p = pearsonr(x, y)
    r_text, p_text = corr_text(r, p)
    correlation_text = f'({panel_label})\n{r_text}\n{p_text}\nn = {len(x)}'
    ax.text(*pos, correlation_text, transform=ax.transAxes, ha='left', color=color)

def add_one_to_one_line(ax, limits, label='1:1', **kwargs):
    return ax.plot(limits, limits, 'k-', alpha=0.75, zorder=0, label=label, **kwargs)[0]

def plot_zero_marker(ax, df, x_col, y_col, label='Zero Values'):
    return ax.scatter(df[x_col], df[y_col], marker='x', color='k', zorder=0, label=label)


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

def remove_samples(df, ignore_isogdgts, ignore_brgdgts):
    brGDGTs = ["Ia", "IIa", "IIa'", "IIIa","IIIa'",
               "Ib", "IIb", "IIb'", "IIIb","IIIb'",
               "Ic", "IIc", "IIc'", "IIIc","IIIc'",]
    isoGDGTs = ["GDGT-0", "GDGT-1", "GDGT-2", "GDGT-3", "GDGT-4", "GDGT-4'"]
    iso_map = (df['Sample Name'].isin(ignore_isogdgts)) & (df['variable'].isin(isoGDGTs))
    br_map  = (df['Sample Name'].isin(ignore_brgdgts))  & (df['variable'].isin(brGDGTs))
    ignored = df.loc[(iso_map)|(br_map)]
    retained = df.loc[~((iso_map)|(br_map))]
    return retained, ignored
    
