"""
Performs a comprehensive, multi-session analysis of unit quality and area mapping.
"""
import sys
import os
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
import json

# Add jnwb to path
sys.path.append(r'D:\jnwb')
from jnwb.oglo import extract_good_units

NWB_DIR = r'D:\OmissionAnalysis\reconstructed_nwbdata'
PROBE_AREA_MAP = {
    'probeA': ['V1', 'V2'],
    'probeB': ['MT', 'MST'],
    'probeC': ['PFC']
}

def map_unit_area(channel_id, probe_name):
    if pd.isna(channel_id): return 'unknown'
    try:
        cid = int(float(channel_id))
    except:
        return 'unknown'
        
    probe_areas = PROBE_AREA_MAP.get(probe_name, [])
    if not probe_areas: return 'unknown'
    
    # Normalize channel ID relative to its probe (0-127)
    norm_cid = cid
    if 'probeB' in probe_name: norm_cid -= 128
    if 'probeC' in probe_name: norm_cid -= 256
        
    if len(probe_areas) == 2:
        # Split probe into two areas
        return probe_areas[0] if norm_cid < 64 else probe_areas[1]
    else:
        # Single area probe
        return probe_areas[0]

def analyze_all_sessions():
    nwb_files = [f for f in os.listdir(NWB_DIR) if f.endswith('.nwb')]
    
    all_sessions_data = []
    
    for filename in nwb_files:
        session_id = filename.split('_')[1].split('-')[1]
        nwb_path = os.path.join(NWB_DIR, filename)
        
        print(f"--- Analyzing Session: {session_id} ---")
        
        try:
            with NWBHDF5IO(nwb_path, 'r', load_namespaces=True) as io:
                nwb = io.read()
                if nwb.units is None: continue

                df_units = nwb.units.to_dataframe()
                df_good_units = extract_good_units(nwb)
                
                def get_area(row):
                    cid = int(float(row['peak_channel_id']))
                    if 0 <= cid < 128: probe = 'probeA'
                    elif 128 <= cid < 256: probe = 'probeB'
                    elif 256 <= cid < 384: probe = 'probeC'
                    else: probe = 'unknown'
                    return map_unit_area(cid, probe)

                df_units['area'] = df_units.apply(get_area, axis=1)
                df_good_units['area'] = df_good_units.apply(get_area, axis=1)
                
                session_stats = {
                    "session_id": session_id,
                    "total_units": len(df_units),
                    "good_units": len(df_good_units),
                    "total_per_area": df_units['area'].value_counts().to_dict(),
                    "good_per_area": df_good_units['area'].value_counts().to_dict()
                }
                all_sessions_data.append(session_stats)
        except Exception as e:
            print(f"Error processing {session_id}: {e}")

    # --- Summary ---
    grand_total_units = sum(d['total_units'] for d in all_sessions_data)
    grand_total_good = sum(d['good_units'] for d in all_sessions_data)
    
    # Aggregate area stats
    area_totals = {}
    area_good = {}
    
    for d in all_sessions_data:
        for area, count in d['total_per_area'].items():
            area_totals[area] = area_totals.get(area, 0) + count
        for area, count in d['good_per_area'].items():
            area_good[area] = area_good.get(area, 0) + count

    # Save to JSON
    with open(r'D:\OmissionAnalysis\all_units_stats.json', 'w') as f:
        json.dump({
            "grand_totals": {"total": grand_total_units, "good": grand_total_good},
            "area_stats": {"total": area_totals, "good": area_good},
            "sessions": all_sessions_data
        }, f, indent=4)
        
    print(f"\nFinal Stats:")
    print(f"Sessions processed: {len(all_sessions_data)}")
    print(f"Grand Total Units: {grand_total_units}")
    print(f"Grand Total Good Units: {grand_total_good}")
    print("\nArea Breakdown (Good Units):")
    for area, count in area_good.items():
        print(f"  {area}: {count}")

if __name__ == "__main__":
    analyze_all_sessions()
