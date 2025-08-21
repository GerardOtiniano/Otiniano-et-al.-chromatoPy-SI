import pandas as pd
import numpy as np

def load_and_melt_manual_data(path):
    df = pd.read_csv(path, encoding='unicode_escape')
    df = df.melt(id_vars=['Sample Name', 'Region'], var_name='variable', value_name='hand_pa')
    return df

def load_and_melt_data_w_unc(path, name='chromatopy'):
    df = pd.read_csv(path)
    df_mean = melt_w_unc(df, name)
    df_low = melt_w_unc(df, name,f'{name}_pa_lower', '_lower_ci')
    df_high = melt_w_unc(df, name,f'{name}_pa_upper', '_upper_ci')
    return df_mean, df_low, df_high

def melt_w_unc(df, name, col_prefix=None, suffix=None):
    if suffix:
        cols = df.columns[df.columns.str.endswith(suffix)]
        temp = df[cols].copy()
        temp['Sample Name'] = df['Sample Name']
        temp.columns = temp.columns.str.replace(suffix, '', regex=False)
        return temp.melt(id_vars=['Sample Name'], var_name='variable', value_name=col_prefix)
    else:
        id_vars = ['Sample Name']
        value_vars = [col for col in df.columns if col in [
            "Ia", "Ib", "Ic", "IIa", "IIb", "IIc", "IIIa", "IIIb", "IIIc",
            "IIa'", "IIb'", "IIc'", "IIIa'", "IIIb'", "IIIc'",
            'GDGT-0', 'GDGT-1', 'GDGT-2', 'GDGT-3', 'GDGT-4', "GDGT-4'"]]
        return df.melt(id_vars=id_vars, value_vars=value_vars, var_name='variable', value_name=f'{name}_pa')


def merge_all(labeled_data, how='inner'):
    """
    Merge multiple labeled datasets (each with main, lower, upper) into a single dataframe.
    
    Parameters:
        labeled_data: list of tuples like (label, df_main, df_lower, df_upper)
    
    Returns:
        Merged dataframe with columns like:
        - label_pa
        - label_pa_lower
        - label_pa_upper
    """
    label0, df_main, df_low, df_high = labeled_data[0]
    df = df_main.copy()
    df = df.rename(columns={'value': f'{label0}_pa'})

    if df_low is not None:
        df = df.merge(df_low.rename(columns={'value': f'{label0}_pa_lower'}), on=['Sample Name', 'variable'], how=how)
    if df_high is not None:
        df = df.merge(df_high.rename(columns={'value': f'{label0}_pa_upper'}), on=['Sample Name', 'variable'], how=how)

    for label, df_main, df_low, df_high in labeled_data[1:]:
        df = df.merge(df_main.rename(columns={'value': f'{label}_pa'}), on=['Sample Name', 'variable'], how=how)
        if df_low is not None:
            df = df.merge(df_low.rename(columns={'value': f'{label}_pa_lower'}), on=['Sample Name', 'variable'], how=how)
        if df_high is not None:
            df = df.merge(df_high.rename(columns={'value': f'{label}_pa_upper'}), on=['Sample Name', 'variable'], how=how)

    return df

def clean_calc_rel_abundance(df):
    df['hand_pa'] = pd.to_numeric(df['hand_pa'], errors='coerce').fillna(0)
    df['hand_ra'] = df['hand_pa'] / df['hand_pa'].max()

    for name in ['chromatopy', 'user_2']:
        # Convert to numeric
        df[f'{name}_pa'] = pd.to_numeric(df[f'{name}_pa'], errors='coerce').fillna(0)
        df[f'{name}_pa_lower'] = pd.to_numeric(df[f'{name}_pa_lower'], errors='coerce').fillna(0)
        df[f'{name}_pa_upper'] = pd.to_numeric(df[f'{name}_pa_upper'], errors='coerce').fillna(0)

        # Convert 95% CI to 1σ
        pa_sigma_lower = df[f'{name}_pa_lower'] / 1.96
        pa_sigma_upper = df[f'{name}_pa_upper'] / 1.96

        # Symmetric sigma = average of upper & lower sigmas
        # to avoid dramatic asymmetry caused by largest peak
        pa_sigma_sym = (pa_sigma_lower + pa_sigma_upper) / 2

        # Find max for scaling
        M_nominal = df[f'{name}_pa'].max()
        M_idx = df[f'{name}_pa'].idxmax()
        M_sigma_sym = pa_sigma_sym[M_idx]

        # Scale values
        df[f'{name}_ra'] = df[f'{name}_pa'] / M_nominal

        # Propagate relative errors (symmetric)
        frac_err = np.sqrt((pa_sigma_sym / df[f'{name}_pa'])**2 +
                           (M_sigma_sym / M_nominal)**2)

        # Store scaled uncertainties using the *_ra_lower / *_ra_upper columns
        df[f'{name}_ra_lower'] = df[f'{name}_ra'] * frac_err
        df[f'{name}_ra_upper'] = df[f'{name}_ra'] * frac_err  # same for symmetric

        # Force zero uncertainty for max peak
        df.loc[M_idx, f'{name}_ra_lower'] = 0
        df.loc[M_idx, f'{name}_ra_upper'] = 0

    return df

def compute_fractional_abundance(df, subset_names, value_col, output_prefix,
                                 propagate_uncertainty=False, lower_col=None, upper_col=None):
    """
    Computes fractional abundances (and optionally asymmetric 1σ errors) for each sample and GDGT subset.

    Expects lower_col / upper_col to be 95% CI half-widths (absolute). Converts to 1σ internally.
    Uses correct propagation for f_i = A_i / T with A_i independent, T = Σ A_k (includes A_i).
    """
    fa_col = output_prefix
    lower_out = f'{output_prefix}_lower'
    upper_out = f'{output_prefix}_upper'
    error_out = f'{output_prefix}_combined_error'

    for sample in df['Sample Name'].unique():
        temp = df[df['Sample Name'] == sample]
        subset = temp[temp['variable'].isin(subset_names)]
        if subset.empty:
            continue

        A = subset[value_col].astype(float).values
        if propagate_uncertainty:
            # 95% CI half-widths -> 1σ
            sig_lo = (subset[lower_col].astype(float).values if lower_col else np.zeros_like(A)) / 1.96
            sig_hi = (subset[upper_col].astype(float).values if upper_col else np.zeros_like(A)) / 1.96
        else:
            sig_lo = sig_hi = np.zeros_like(A)

        T = np.nansum(A)
        if not np.isfinite(T) or T <= 0:
            # write NaNs / zeros and continue
            for var in subset_names:
                condition = (df['Sample Name'] == sample) & (df['variable'] == var)
                df.loc[condition, [fa_col, lower_out, upper_out, error_out]] = np.nan
            continue

        S_lo = np.nansum(sig_lo**2)
        S_hi = np.nansum(sig_hi**2)

        for idx, var in enumerate(subset['variable'].values):
            Ai = A[idx]
            fi = Ai / T if Ai > 0 else (0.0 if T > 0 else np.nan)

            condition = (df['Sample Name'] == sample) & (df['variable'] == var)
            df.loc[condition, fa_col] = fi

            if propagate_uncertainty and Ai > 0:
                # variance for lower side (use lower sigmas everywhere)
                v_lo = ((T - Ai)**2 * (sig_lo[idx]**2) + (Ai**2) * (S_lo - sig_lo[idx]**2)) / (T**4)
                # variance for upper side (use upper sigmas everywhere)
                v_hi = ((T - Ai)**2 * (sig_hi[idx]**2) + (Ai**2) * (S_hi - sig_hi[idx]**2)) / (T**4)

                abs_err_lower = np.sqrt(max(v_lo, 0.0))
                abs_err_upper = np.sqrt(max(v_hi, 0.0))
                df.loc[condition, lower_out] = abs_err_lower
                df.loc[condition, upper_out] = abs_err_upper
                df.loc[condition, error_out] = 0.5 * (abs_err_lower + abs_err_upper)

    return df


def remove_mis_id(df, misid_names):
    brgdgts = ["Ia", "IIa", "IIa'", "IIIa", "IIIa'", "Ib", "IIb", "IIb'", "IIIb", "IIIb'", "Ic", "IIc", "IIc'", "IIIc", "IIIc'"]
    mis_id = df[df['Sample Name'].isin(misid_names)]
    df = df.loc[~((df['Sample Name'].isin(misid_names))&(df['variable'].isin(brgdgts)))]
    return df, mis_id