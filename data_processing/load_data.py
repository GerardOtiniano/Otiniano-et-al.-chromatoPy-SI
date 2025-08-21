import pandas as pd
import numpy as np
from data_processing.process_data_functions import *

def load_chraomtopy_and_manual(ignore_misidentified = True):
    # Load datasets
    manual_df = load_and_melt_manual_data('data/manual_peak_areas.csv')
    chroma, chroma_low, chroma_high = load_and_melt_data_w_unc('data/chromatoPy_peak_areas.csv')
    user_2, user_2_low, user_2_high = load_and_melt_data_w_unc('data/user_2_peak_areas.csv', name='user_2')

    # merge melted datasets
    df = merge_all([("manual", manual_df, None, None), 
                    ("chromatopy", chroma, chroma_low, chroma_high),
                    ("user_2", user_2, user_2_low, user_2_high)])
    ignore = ['H1608000196', 'H1608000191', 'H1608000189', 'H2102002', 'H2102003', 'H2102004']
    df, df_sans_mis_id = remove_mis_id(df, ignore)
    temp = df.loc[~((df.chromatopy_pa==0) & (df.hand_pa == 0))]
    # calculate realtive abundance (w.r.t. largest peak)
    df = clean_calc_rel_abundance(df)

    # calculate fractional abundance
    branched = ["Ia", "IIa", "IIa'", "IIIa", "IIIa'", "Ib", "IIb", "IIb'", "IIIb", "IIIb'", "Ic", "IIc", "IIc'", "IIIc", "IIIc'"]
    isoprenoid = ['GDGT-0', 'GDGT-1', 'GDGT-2', 'GDGT-3', 'GDGT-4', "GDGT-4'"]

        # chromatoPy
    df = compute_fractional_abundance(df, branched, value_col='chromatopy_pa', 
                                      output_prefix='chromatopy_fa', propagate_uncertainty=True,
                                      lower_col='chromatopy_pa_lower', upper_col='chromatopy_pa_upper')
    df = compute_fractional_abundance(df, isoprenoid, value_col='chromatopy_pa', 
                                      output_prefix='chromatopy_fa', propagate_uncertainty=True,
                                      lower_col='chromatopy_pa_lower', upper_col='chromatopy_pa_upper')
        # manual 
    df = compute_fractional_abundance(df, branched, value_col='hand_pa', 
                                      output_prefix='hand_fa')
    df = compute_fractional_abundance(df, isoprenoid, value_col='hand_pa', 
                                      output_prefix='hand_fa')
    
        # user_2 
    df = compute_fractional_abundance(df, branched, value_col='user_2_pa', 
                                      output_prefix='user_2_fa', propagate_uncertainty=True,
                                      lower_col='user_2_pa_lower', upper_col='user_2_pa_upper')
    df = compute_fractional_abundance(df, isoprenoid, value_col='user_2_pa',
                                      output_prefix='user_2_fa', propagate_uncertainty=True,
                                      lower_col='user_2_pa_lower', upper_col='user_2_pa_upper')
    # Remove brGDGTs samples not processed for brGDGTs manually
    ignore = ['H1608000196', 'H1608000191', 'H1608000189', 'H2102002', 'H2102003', 'H2102004']
    df, df_sans_mis_id = remove_mis_id(df, ignore)
    return df.fillna(0)


