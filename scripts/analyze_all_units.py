"""
Performs a comprehensive, multi-session analysis of unit quality and area mapping.
Filters sessions for 'omission_glo_passive' task availability.
Dynamically extracts area mapping from electrode metadata.
"""
import sys
import os
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
import json
import gc

# Add jnwb to path
sys.path.append(r'D:\jnwb')
from jnwb.oglo import extract_good_units

NWB_DIR = r'D:\OmissionAnalysis\reconstructed_nwbdata'

def get_session_area_mapping(nwb):
    """
    Dynamically extracts probe-to-area mapping from the electrodes table.
    Returns a dict: {probe_name: [list of areas]}
    """
    df_elec = nwb.electrodes.to_dataframe()
    probe_col = 'group_name' if 'group_name' in df_elec.columns else 'probe'
    
    mapping = {}
    if probe_col in df_elec.columns:
        for probe, group in df_elec.groupby(probe_col):
            unique_entries = group['location'].unique().tolist()
            
            # Split entries that contain commas or slashes and flatten the list
            all_areas = []
            for entry in unique_entries:
                if pd.isna(entry) or entry.lower() in ['unknown', '']:
                    continue
                # Split by comma or slash
                parts = [p.strip() for p in entry.replace('/', ',').split(',')]
                for p in parts:
                    if p and p not in all_areas:
                        all_areas.append(p)
            
            mapping[probe] = all_areas
            
    return mapping

def map_unit_to_area_v2(row, session_mapping):
    """
    Maps a unit to a brain area based on peak_channel_id and the session's dynamic mapping.
    Applies the rule: divide the 128 channels of a probe equally among its areas.
    """
    cid_raw = int(float(row['peak_channel_id']))
    
    # Determine probe name and local channel index (0-127)
    if 0 <= cid_raw < 128:
        probe = 'probeA'
        cid_local = cid_raw
    elif 128 <= cid_raw < 256:
        probe = 'probeB'
        cid_local = cid_raw - 128
    elif 256 <= cid_raw < 384:
        probe = 'probeC'
        cid_local = cid_raw - 256
    else:
        return 'unknown'
        
    areas = session_mapping.get(probe, [])
    if not areas:
        return 'unknown'
    
    n_areas = len(areas)
    if n_areas == 1:
        return areas[0]
    
    # Divide the 128 channels equally
    # e.g., if 2 areas: 0-63 is areas[0], 64-127 is areas[1]
    # index = floor(cid_local / (128 / n_areas))
    area_idx = int(cid_local // (128 / n_areas))
    # Cap index to handle edge cases
    area_idx = min(area_idx, n_areas - 1)
    
    return areas[area_idx]

def analyze_all_sessions():
    nwb_files = [f for f in os.listdir(NWB_DIR) if f.endswith('.nwb')]
    
    all_sessions_data = []
    
    for filename in nwb_files:
        session_id = filename.split('_')[1].split('-')[1]
        nwb_path = os.path.join(NWB_DIR, filename)
        
        try:
            with NWBHDF5IO(nwb_path, 'r', load_namespaces=True) as io:
                nwb = io.read()
                
                if 'omission_glo_passive' not in nwb.intervals:
                    print(f"Skipping {session_id}: 'omission_glo_passive' not found.")
                    continue

                if nwb.units is None: continue

                df_units = nwb.units.to_dataframe()
                
                # Get session-specific mapping
                session_mapping = get_session_area_mapping(nwb)
                print(f"  {session_id} Mapping: {session_mapping}")

                # Define Good Units: Quality=1 OR PR > 0.9
                mask_good = (df_units.index != df_units.index) # All False
                if 'quality' in df_units.columns:
                    mask_good |= (df_units['quality'].astype(str).isin(['1.0', '1', 'good']))
                if 'presence_ratio' in df_units.columns:
                    mask_good |= (pd.to_numeric(df_units['presence_ratio'], errors='coerce') > 0.9)
                
                df_units['is_good'] = mask_good
                # Pass session_mapping to the mapper
                df_units['area'] = df_units.apply(lambda r: map_unit_to_area_v2(r, session_mapping), axis=1)
                
                df_good = df_units[df_units['is_good']]
                
                session_stats = {
                    "session_id": session_id,
                    "total_units": len(df_units),
                    "good_units": len(df_good),
                    "total_per_area": df_units['area'].value_counts().to_dict(),
                    "good_per_area": df_good['area'].value_counts().to_dict(),
                    "probes": list(session_mapping.keys())
                }
                all_sessions_data.append(session_stats)
                print(f"  Processed {session_id}: {len(df_good)} good units.")
        except Exception as e:
            print(f"Error processing {session_id}: {e}")

    # --- Generate Statistics ---
    grand_total_units = sum(d['total_units'] for d in all_sessions_data)
    grand_total_good = sum(d['good_units'] for d in all_sessions_data)
    
    area_totals = {}
    area_good = {}
    for d in all_sessions_data:
        for area, count in d['total_per_area'].items():
            area_totals[area] = area_totals.get(area, 0) + count
        for area, count in d['good_per_area'].items():
            area_good[area] = area_good.get(area, 0) + count

    # --- Construct Markdown Report ---
    lines = []
    lines.append("# jnwb Master Unit Statistics & Area Mapping")
    lines.append("\nThis report provides a dynamic overview of neural quality across the Omission dataset, with area mapping extracted from electrode metadata.")
    lines.append("\n## 📊 Grand Totals")
    lines.append("| Category | Count |")
    lines.append("| :--- | :---: |")
    lines.append(f"| **Total Sessions** | {len(all_sessions_data)} |")
    lines.append(f"| **Total Neurons Recorded** | {grand_total_units} |")
    lines.append(f"| **High-Quality Neurons** | <span style='color:#CFB87C;'>**{grand_total_good}**</span> |")
    
    lines.append("\n## 🧠 Distribution by Brain Area (All Sessions)")
    lines.append("| Area | Total Units | Good Units | Stability (%) |")
    lines.append("| :--- | :---: | :---: | :---: |")
    for area in sorted(area_totals.keys()):
        total = area_totals[area]
        good = area_good.get(area, 0)
        stab = (good / total * 100) if total > 0 else 0
        lines.append(f"| **{area}** | {total} | {good} | {stab:.1f}% |")

    lines.append("\n## 📅 Per-Session Detail")
    lines.append("| Session ID | Good Units | Probes | Areas (Mapped) |")
    lines.append("| :--- | :---: | :--- | :--- |")
    
    for d in all_sessions_data:
        areas_str = ", ".join(sorted(d['good_per_area'].keys()))
        probes_str = ", ".join(sorted(d['probes']))
        lines.append(f"| **{d['session_id']}** | {d['good_units']} | {probes_str} | {areas_str} |")

    report = "\n".join(lines)
    
    output_path = r'D:\jnwb\UNIT_STATS_SUMMARY.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nFinal Statistics saved to {output_path}")

if __name__ == "__main__":
    analyze_all_sessions()
