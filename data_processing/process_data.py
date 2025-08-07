import pandas as pd
import numpy as np
from process_data_functions import *

"""
Code to process chromatoPy output for use in analysis and figures for Otiniano et al.
"""

####################################################
#################### Import data ###################
####################################################
manual_data = pd.read_csv('data/manual_integration_GDGTs.csv', encoding='unicode_escape')
regions = manual_data[['Sample Name', 'Region']]
manual_data = manual_data.melt(id_vars=['Sample Name', 'Region'])
manual_data = manual_data.rename(columns={'value': 'hand_pa'})

# Import chroamtopy data
chromatopy_data = pd.read_csv('data/chromatoPy_integration_GDGTs.csv')
columns_of_interest = ["Sample Name", "Ia", "Ib", "Ic", "IIa", "IIb", "IIc", "IIIa", "IIIb", "IIIc", "IIa'", "IIb'", "IIc'", "IIIa'", "IIIb'", "IIIc'", 'GDGT-0', 'GDGT-1', 'GDGT-2', 'GDGT-3', 'GDGT-4', "GDGT-4'"]
chpy = chromatopy_data[columns_of_interest]
chpy = chpy.melt(id_vars=['Sample Name'], value_vars = columns_of_interest)
# chpy = chpy.melt(id_vars='Sample Name')
chpy = chpy.rename(columns={'value': 'chromatopy_pa'})

####################################################
################### Process data ###################
####################################################

# Prepare uncertainty columns
chpy_upper = chromatopy_data.iloc[:, chromatopy_data.columns.str.contains('upper')].copy()
chpy_upper.loc[:, 'Sample Name'] = chpy.loc[:, 'Sample Name']
chpy_upper.columns = chpy_upper.columns.str.replace('_upper_ci', '')
chpy_upper = chpy_upper.melt(id_vars=['Sample Name'])
chpy_upper = chpy_upper.rename(columns={'value': 'chromatopy_pa_upper'})

chpy_lower = chromatopy_data.iloc[:, chromatopy_data.columns.str.contains('lower')].copy()
chpy_lower.loc[:, 'Sample Name'] = chpy.loc[:, 'Sample Name']
chpy_lower.columns = chpy_lower.columns.str.replace('_lower_ci', '')
chpy_lower = chpy_lower.melt(id_vars=['Sample Name'])
chpy_lower = chpy_lower.rename(columns={'value': 'chromatopy_pa_lower'})

# Merge Data
df = pd.merge(manual_data, chpy, on=['Sample Name', 'variable'], how='inner')
df = pd.merge(df, chpy_lower, on=['Sample Name', 'variable'], how='inner')
df = pd.merge(df, chpy_upper, on=['Sample Name', 'variable'], how='inner')
df['hand_pa'] = pd.to_numeric(df['hand_pa'], errors='coerce')
df['hand_pa'] = df['hand_pa'].fillna(0)
df['chromatopy_pa'] = pd.to_numeric(df['chromatopy_pa'], errors='coerce')
df['chromatopy_pa'] = df['chromatopy_pa'].fillna(0)
df['chromatopy_pa_lower'] = df['chromatopy_pa']-df['chromatopy_pa_lower']
df['chromatopy_pa_upper'] = df['chromatopy_pa_upper']-df['chromatopy_pa']

# Calculate relative abundance (relative to maximum peak area accross all samples)
df['hand_ra']=df['hand_pa']/df['hand_pa'].max()
M_nominal = df['chromatopy_pa'].max()
df['chromatopy_ra']=df['chromatopy_pa']/M_nominal

# Propogate uncertainty
M_lower = (df[df['chromatopy_pa']==M_nominal]['chromatopy_pa_lower'].iloc[0])
M_upper = (df[df['chromatopy_pa']==M_nominal]['chromatopy_pa_upper'].iloc[0])
df['chromatopy_ra_lower'] = df['chromatopy_ra']*np.sqrt(((df['chromatopy_pa_lower']/df['chromatopy_pa'])**2))+(M_lower/M_nominal)**2
df['chromatopy_ra_upper'] = df['chromatopy_ra']*np.sqrt(((df['chromatopy_pa_upper']/df['chromatopy_pa'])**2))+(M_upper/M_nominal)**2

####################################################
#### Calculate fractional abundance (per sample) ###
####################################################
branched = ["Ia", "IIa", "IIa'", "IIIa", "IIIa'",
            "Ib", "IIb", "IIb'", "IIIb", "IIIb'",
            "Ic", "IIc", "IIc'", "IIIc", "IIIc'"]
isoprenoid = ['GDGT-0', 'GDGT-1', 'GDGT-2', 'GDGT-3', 'GDGT-4', "GDGT-4'"]

# Ensure the result columns exist
df['chromatopy_fa'] = np.nan
df['chromatopy_fa_lower'] = np.nan
df['chromatopy_fa_upper'] = np.nan
df['chromatopy_fa_combined_error'] = np.nan

def fractional_abundance_w_error(subset_names):
    for sample in df['Sample Name'].unique():
        temp = df[df['Sample Name'] == sample]
        subset = temp[temp['variable'].isin(subset_names)]
        if subset.empty:
            continue  # Skip if there are no measurements for this subset
        
        # Compute the total (denominator) and propagate errors by summing in quadrature
        T = subset['chromatopy_pa'].sum()
        T_lower = np.sqrt((subset['chromatopy_pa_lower']**2).sum())
        T_upper = np.sqrt((subset['chromatopy_pa_upper']**2).sum())
        
        # Loop through each target variable in the subset
        for var in subset_names:
            if var not in subset['variable'].values:
                continue
            # Assuming one measurement per (sample, variable)
            row = subset[subset['variable'] == var].iloc[0]
            A = row['chromatopy_pa']
            A_lower = row['chromatopy_pa_lower']
            A_upper = row['chromatopy_pa_upper']
            
            # Compute the fractional abundance
            f = A / T
            # Propagate the error for division:
            f_err_lower = np.sqrt(((A_lower/A)**2)+(T_lower/T)**2)
            f_err_upper = np.sqrt(((A_upper/A)**2)+(T_upper/T)**2)
            # Combine the two uncertainties into a single symmetric uncertainty (e.g., the arithmetic mean)
            f_err_combined = (f_err_lower + f_err_upper) / 2
            condition = (df['Sample Name'] == sample) & (df['variable'] == var)
            df.loc[condition, 'chromatopy_fa'] = f
            df.loc[condition, 'chromatopy_fa_lower'] = f_err_lower
            df.loc[condition, 'chromatopy_fa_upper'] = f_err_upper
            df.loc[condition, 'chromatopy_fa_combined_error'] = f_err_combined
            
fractional_abundance_w_error(branched)   # For branched measurements
fractional_abundance_w_error(isoprenoid)  # For isoprenoid measurements

def fractional_abundance_manual(subset_names):
    for sample in df['Sample Name'].unique():
        temp = df[df['Sample Name'] == sample]
        subset = temp[temp['variable'].isin(subset_names)]
        if subset.empty:
            continue  # Skip if there are no measurements for this subset
        T = subset['hand_pa'].sum()
        for var in subset_names:
            if var not in subset['variable'].values:
                continue
            row = subset[subset['variable'] == var].iloc[0]
            A = row['hand_pa']
            f = A / T
            condition = (df['Sample Name'] == sample) & (df['variable'] == var)
            df.loc[condition, 'hand_fa'] = f
fractional_abundance_manual(branched)
fractional_abundance_manual(isoprenoid)

df.to_csv('data/chromatoPy and manual data.csv')

# %%
import pandas as pd
import numpy as np
from process_data_functions import *

manual_df = load_manual_data('data/manual_integration_GDGTs.csv')
chroma = load_chroma_data('data/chromatoPy_integration_GDGTs.csv')

chroma_melted = melt_chromatopy(chroma)
lower_melted = melt_chromatopy(chroma, 'chromatopy_pa_lower', '_lower_ci')
upper_melted = melt_chromatopy(chroma, 'chromatopy_pa_upper', '_upper_ci')

df = merge_all(manual_df, chroma_melted, lower_melted, upper_melted)
df = clean_and_calc_rel_abundance(df)

# branched = ["Ia", "IIa", "IIa'", "IIIa", "IIIa'", "Ib", "IIb", "IIb'", "IIIb", "IIIb'", "Ic", "IIc", "IIc'", "IIIc", "IIIc'"]
# isoprenoid = ['GDGT-0', 'GDGT-1', 'GDGT-2', 'GDGT-3', 'GDGT-4', "GDGT-4'"]

# fractional_abundance_w_error(df,branched)
# fractional_abundance_w_error(df,isoprenoid)
# fractional_abundance_manual(df,branched)
# fractional_abundance_manual(df,isoprenoid)
branched = ["Ia", "IIa", "IIa'", "IIIa", "IIIa'", "Ib", "IIb", "IIb'", "IIIb", "IIIb'", "Ic", "IIc", "IIc'", "IIIc", "IIIc'"]
isoprenoid = ['GDGT-0', 'GDGT-1', 'GDGT-2', 'GDGT-3', 'GDGT-4', "GDGT-4'"]

# Fractional abundance - chromatoPy
df = compute_fractional_abundance(df, branched, value_col='chromatopy_pa', 
                                  output_prefix='chromatopy_fa', propagate_uncertainty=True,
                                  lower_col='chromatopy_pa_lower', upper_col='chromatopy_pa_upper')
df = compute_fractional_abundance(df, isoprenoid, value_col='chromatopy_pa', 
                                  output_prefix='chromatopy_fa', propagate_uncertainty=True,
                                  lower_col='chromatopy_pa_lower', upper_col='chromatopy_pa_upper')

# Fractional abundance - manual 
df = compute_fractional_abundance(df, branched, value_col='hand_pa', 
                                  output_prefix='hand_fa')
df = compute_fractional_abundance(df, isoprenoid, value_col='hand_pa', 
                                  output_prefix='hand_fa')

df.to_csv('data/test.csv', index=False, na_rep='NaN')