"""
oglo_v2.py: A precise Python-based replication of the legacy MATLAB
logic for identifying the 12 Visual Omission Oddball Paradigm trial groups.
"""
import pandas as pd
import numpy as np

def get_oglo_trial_masks_v2(df: pd.DataFrame) -> dict:
    """
    Precisely identifies the 12 OGLO trial groups from an NWB interval dataframe,
    based on the exact logic in the legacy `jOGLOSignals.m` script.

    Args:
        df (pd.DataFrame): DataFrame from nwb.intervals['omission_glo_passive'].
                           Requires 'task_condition_number', 'correct', 'stimulus_number'.

    Returns:
        dict: A dictionary mapping condition names to boolean pandas Series masks.
    """
    df['task_condition_number'] = pd.to_numeric(df['task_condition_number'], errors='coerce').fillna(0)
    df['stimulus_number'] = pd.to_numeric(df['stimulus_number'], errors='coerce').fillna(0)
    
    # Base filter for correct trials
    base_mask = (df['correct'].astype(str) == '1.0')
    
    conditions = df['task_condition_number']
    
    # Define the condition lists from MATLAB
    conditionlist_rrxr = [35, 37, 39, 41]
    conditionlist_rrrx = [36, 38, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    
    # Replicate the logic from the MATLAB `parfor` loop
    masks = {
        'AAAB': base_mask & (conditions > 0) & (conditions <= 2),
        'AXAB': base_mask & (conditions > 2) & (conditions <= 3),
        'AAXB': base_mask & (conditions > 3) & (conditions <= 4),
        'AAAX': base_mask & (conditions > 4) & (conditions <= 5),
        'BBBA': base_mask & (conditions > 5) & (conditions <= 7),
        'BXBA': base_mask & (conditions > 7) & (conditions <= 8),
        'BBXA': base_mask & (conditions > 8) & (conditions <= 9),
        'BBBX': base_mask & (conditions > 9) & (conditions <= 10),
        'RRRR': base_mask & (conditions > 10) & (conditions <= 26), # This corresponds to RRRR Standard
        'RXRR': base_mask & (conditions > 26) & (conditions <= 34), # This corresponds to RXRR
        'RRXR': base_mask & conditions.isin(conditionlist_rrxr),
        'RRRX': base_mask & conditions.isin(conditionlist_rrrx)
    }
    
    return masks
