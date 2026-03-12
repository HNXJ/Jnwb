"""
run_information_dynamics_master.py: Production suite for Omission Information and Variation.
Features: GPU-Accelerated RF Ranking, SEM patches, Multi-Condition comparisons.
"""
import os
import h5py
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import sem
import gc
import sys

# Add jnwb to path
sys.path.append(r'D:\jnwb')
from jnwb.analysis import BANDS, compute_trial_tfr_dynamics, compute_variability_quenching

# Configuration
DATA_DIR = r'D:\OmissionAnalysis'
FIGURES_DIR = r'D:\OmissionAnalysis\figures'
FS = 1000

def plot_dynamics_comparison(fig, time, m1, s1, name1, m2, s2, name2, title, out_path):
    fig.add_trace(go.Scatter(x=time, y=m1, name=name1, line=dict(color="#CFB87C")))
    fig.add_trace(go.Scatter(x=np.concatenate([time, time[::-1]]), y=np.concatenate([m1+s1, (m1-s1)[::-1]]), fill='toself', fillcolor="#CFB87C", opacity=0.2, showlegend=False))
    
    fig.add_trace(go.Scatter(x=time, y=m2, name=name2, line=dict(color="cyan")))
    fig.add_trace(go.Scatter(x=np.concatenate([time, time[::-1]]), y=np.concatenate([m2+s2, (m2-s2)[::-1]]), fill='toself', fillcolor="cyan", opacity=0.2, showlegend=False))
    
    fig.update_layout(title=title, template="plotly_dark", xaxis_title="Time (ms)")
    fig.write_html(out_path)

def rank_channels(X, y, channel_names):
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    return [(channel_names[i], importances[i]) for i in indices]

def run_master_analysis(session_id):
    print(f"\n>>> Master Analysis: {session_id}")
    lfp_path = os.path.join(DATA_DIR, f'lfp_by_area_ses-{session_id}.h5')
    if not os.path.exists(lfp_path): return

    with h5py.File(lfp_path, 'r') as hl:
        areas = sorted(list(hl.keys()))
        time_ms = np.linspace(-1000, 5000, 6000)
        
        for area in areas:
            if 'AAAX' not in hl[area] or 'RXRR' not in hl[area]: continue
            
            # 1. Expanded Dynamics (Gamma example)
            m_om, s_om = compute_trial_tfr_dynamics(hl[f"{area}/AAAX"][()], FS, 'gamma')
            m_rand, s_rand = compute_trial_tfr_dynamics(hl[f"{area}/RXRR"][()], FS, 'gamma')
            
            fig = go.Figure()
            plot_dynamics_comparison(fig, time_ms, m_om, s_om, "Omission", m_rand, s_rand, "Random", f"{session_id} {area} Gamma Dynamics", os.path.join(FIGURES_DIR, f"ses-{session_id}_{area}_gamma_master.html"))

            # 2. Variability Quenching
            v_om = compute_variability_quenching(hl[f"{area}/AAAX"][()])
            # (Just saving one quenching plot per area for brevity)
            fig_v = go.Figure(data=go.Scatter(x=time_ms, y=v_om, name="Omission Variation", line=dict(color="#CFB87C")))
            fig_v.update_layout(title=f"{session_id} {area} Variability Quenching", template="plotly_dark")
            fig_v.write_html(os.path.join(FIGURES_DIR, f"ses-{session_id}_{area}_quenching.html"))

            # 3. Information Ranking
            # Feature: Mean firing in 3000-4000ms window
            data_om = hl[f"{area}/AAAX"][()][:, :, 4000:5000].mean(axis=2)
            data_rand = hl[f"{area}/RXRR"][()][:, :, 4000:5000].mean(axis=2)
            
            X = np.vstack([data_om, data_rand])
            y = np.concatenate([np.ones(len(data_om)), np.zeros(len(data_rand))])
            
            top_10 = rank_channels(X, y, [f"Ch_{i}" for i in range(X.shape[1])])
            print(f"    Top Channels for {area}: {top_10[:3]}")

    gc.collect()

def main():
    lfp_files = [f for f in os.listdir(DATA_DIR) if f.startswith('lfp_by_area') and f.endswith('.h5')]
    for f in lfp_files:
        sid = f.split('_')[-1].split('.')[0].replace('ses-', '')
        run_master_analysis(sid)

if __name__ == "__main__":
    main()
