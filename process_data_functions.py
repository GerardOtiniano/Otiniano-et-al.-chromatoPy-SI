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

def clean_and_calc_rel_abundance(df):
    df['hand_pa'] = pd.to_numeric(df['hand_pa'], errors='coerce').fillna(0)
    df['hand_ra'] = df['hand_pa'] / df['hand_pa'].max()
    for name in ['chromatopy', 'user_2']:
        df[f'{name}_pa'] = pd.to_numeric(df[f'{name}_pa'], errors='coerce').fillna(0)
        df[f'{name}_pa_lower'] = df[f'{name}_pa'] - df[f'{name}_pa_lower']
        df[f'{name}_pa_upper'] = df[f'{name}_pa_upper'] - df[f'{name}_pa']
        M_nominal = df[f'{name}_pa'].max()
        df[f'{name}_ra'] = df[f'{name}_pa'] / M_nominal
        M_lower = df.loc[df[f'{name}_pa'] == M_nominal, f'{name}_pa_lower'].iloc[0]
        M_upper = df.loc[df[f'{name}_pa'] == M_nominal, f'{name}_pa_upper'].iloc[0]
        df[f'{name}_ra_lower'] = df[f'{name}_ra'] * np.sqrt(
            (df[f'{name}_pa_lower'] / df[f'{name}_pa']) ** 2 + (M_lower / M_nominal) ** 2)
        df[f'{name}_ra_upper'] = df[f'{name}_ra'] * np.sqrt(
            (df[f'{name}_pa_upper'] / df[f'{name}_pa']) ** 2 + (M_upper / M_nominal) ** 2)
    return df

def compute_fractional_abundance(df, subset_names, value_col, output_prefix,
                                 propagate_uncertainty=False, lower_col=None, upper_col=None):
    """
    Computes fractional abundances (and optionally errors) for each sample and GDGT variable.

    Parameters:
        df: DataFrame containing GDGT data.
        subset_names: List of GDGTs to consider (branched, isoprenoid, etc.)
        value_col: Column name for peak area values.
        output_prefix: Prefix for output columns (e.g., 'chromatopy_fa' or 'hand_fa').
        propagate_uncertainty: Whether to compute error propagation.
        lower_col: Name of lower uncertainty column (required if propagate_uncertainty).
        upper_col: Name of upper uncertainty column (required if propagate_uncertainty).
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

        T = subset[value_col].sum()
        T_lower = np.sqrt((subset[lower_col] ** 2).sum()) if propagate_uncertainty else None
        T_upper = np.sqrt((subset[upper_col] ** 2).sum()) if propagate_uncertainty else None

        for var in subset_names:
            row = subset[subset['variable'] == var]
            if row.empty:
                continue

            A = row[value_col].values[0]
            if A>0 and T>0:
                f = A / T
            else:
                f = np.nan

            condition = (df['Sample Name'] == sample) & (df['variable'] == var)
            df.loc[condition, fa_col] = f

            if propagate_uncertainty and A > 0:
                A_lower = row[lower_col].values[0]
                A_upper = row[upper_col].values[0]
                f_err_lower = np.sqrt((A_lower / A) ** 2 + (T_lower / T) ** 2)
                f_err_upper = np.sqrt((A_upper / A) ** 2 + (T_upper / T) ** 2)
                f_err_combined = (f_err_lower + f_err_upper) / 2
                df.loc[condition, lower_out] = f_err_lower
                df.loc[condition, upper_out] = f_err_upper
                df.loc[condition, error_out] = f_err_combined
    return df

def remove_mis_id(df, misid_names):
    brgdgts = ["Ia", "IIa", "IIa'", "IIIa", "IIIa'", "Ib", "IIb", "IIb'", "IIIb", "IIIb'", "Ic", "IIc", "IIc'", "IIIc", "IIIc'"]
    mis_id = df[df['Sample Name'].isin(misid_names)]
    df = df.loc[~((df['Sample Name'].isin(misid_names))&(df['variable'].isin(brgdgts)))]
    df=df.fillna(0)
    return df, mis_id