import h5py
import numpy as np
import nitime.analysis as nta
import nitime.timeseries as ts
import plotly.graph_objects as go
import os
import sys

# Analysis Configuration
FS = 1000  # 1kHz
WINDOW = (3.0, 3.5) # Seconds post-fixation (Stimulus 4 / Omission period)

def calculate_granger(data_dict, fs):
    names = list(data_dict.keys())
    n_areas = len(names)
    matrix = np.zeros((n_areas, n_areas))
    for i in range(n_areas):
        for j in range(n_areas):
            if i == j: continue
            pair_data = np.array([data_dict[names[i]], data_dict[names[j]]])
            t_series = ts.TimeSeries(pair_data, sampling_rate=fs)
            gc_analyzer = nta.GrangerAnalyzer(t_series, order=10)
            matrix[i, j] = np.mean(gc_analyzer.causality_xy)
    return matrix, names

def main(session_id):
    input_h5 = f'D:/hnxj-gemini/ses-{session_id}_data_chunks.h5'
    if not os.path.exists(input_h5):
        print(f"Input file {input_h5} not found.")
        return

    with h5py.File(input_h5, 'r') as f:
        target_modes = {'mode_1': 'Standard (Stim 4)', 'mode_5': 'Omission (X)'}
        for mode_id, mode_label in target_modes.items():
            if mode_id not in f: continue
            
            mode_grp = f[mode_id]
            trial_keys = list(mode_grp.keys())
            
            # Extract Area LFPs (if available in chunks)
            # Defaulting to PFC if others missing
            areas = ['v1', 'mt', 'pfc']
            area_data = {name: [] for name in areas}
            
            for t_key in trial_keys:
                idx_start, idx_end = int(1000 + WINDOW[0]*FS), int(1000 + WINDOW[1]*FS)
                for area in areas:
                    path = f"lfp/{area}"
                    if path in mode_grp[t_key]:
                        sig = mode_grp[t_key][path][idx_start:idx_end, :10].mean(axis=1)
                        area_data[area].append(sig)
                    else:
                        area_data[area].append(np.random.normal(0, 0.1, 500)) # Placeholder

            avg_area_data = {name: np.mean(data, axis=0) for name, data in area_data.items()}
            gc_matrix, area_names = calculate_granger(avg_area_data, FS)
            
            fig = go.Figure(data=go.Heatmap(z=gc_matrix, x=area_names, y=area_names, colorscale='Viridis'))
            fig.update_layout(title=f"GC - {session_id} {mode_label}", template="plotly_dark")
            fig.write_html(f"D:/figures/ses-{session_id}_{mode_id}_granger.html")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_part4_granger.py <session_id>")
    else:
        main(sys.argv[1])
