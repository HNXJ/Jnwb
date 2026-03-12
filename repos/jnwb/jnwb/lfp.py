"""
lfp.py: High-efficiency LFP extraction and area mapping.
Part of the jnwb package.
"""
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
import os

def get_lfp_probe_mapping(nwb):
    """
    Extracts probe-to-area mapping from electrodes table.
    Identifies which probe (A, B, C) corresponds to which areas.
    """
    df_elec = nwb.electrodes.to_dataframe()
    probe_col = 'group_name' if 'group_name' in df_elec.columns else 'probe'
    
    mapping = {}
    if probe_col in df_elec.columns:
        for probe, group in df_elec.groupby(probe_col):
            unique_areas = [a for a in group['location'].unique().tolist() if a.lower() not in ['unknown', '']]
            mapping[probe] = unique_areas
    return mapping

def map_channels_to_areas(probe_mapping):
    """
    Creates a direct channel-to-area lookup map.
    Assumes 128 channels per probe (A: 0-127, B: 128-255, C: 256-383).
    """
    channel_map = {}
    
    # Standard probe offsets
    offsets = {'probeA': 0, 'probeB': 128, 'probeC': 256}
    
    for probe, areas in probe_mapping.items():
        if probe not in offsets: continue
        offset = offsets[probe]
        n_areas = len(areas)
        
        if n_areas == 1:
            for c in range(128):
                channel_map[offset + c] = areas[0]
        elif n_areas >= 2:
            # Divide probe equally
            chunk = 128 // n_areas
            for i, area in enumerate(areas):
                start = i * chunk
                end = (i + 1) * chunk if i < n_areas - 1 else 128
                for c in range(start, end):
                    channel_map[offset + c] = area
                    
    return channel_map

def extract_lfp_epoch(lfp_series, start_time, duration_s):
    """
    Lazily extracts an LFP slice from an ElectricalSeries.
    """
    fs = lfp_series.rate
    t0 = lfp_series.starting_time if lfp_series.starting_time is not None else 0.0
    
    start_idx = int((start_time - t0) * fs)
    n_samples = int(duration_s * fs)
    
    if start_idx < 0: return None
    
    try:
        # Slice only what we need to save RAM
        return lfp_series.data[start_idx : start_idx + n_samples]
    except:
        return None
