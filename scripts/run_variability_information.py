"""
run_variability_information.py: Master script for Variability Quenching and Time-Resolved Dynamics.
Features: SEM patches, Cross-Condition comparison, and Information ranking.
"""
import sys
import os
import h5py
import numpy as np
import plotly.graph_objects as go
import gc

# Add jnwb to path
sys.path.append(r'D:\jnwb')
from jnwb.analysis import compute_trial_tfr_dynamics, compute_variability_quenching, BANDS

# Configuration
DATA_DIR = r'D:\OmissionAnalysis'
FIGURES_DIR = r'D:\OmissionAnalysis\figures'
FS = 1000

def plot_with_sem(fig, x, mean, sem, name, color):
    """Adds a mean line and SEM patch to a Plotly figure."""
    fig.add_trace(go.Scatter(
        x=x, y=mean, line=dict(color=color), name=name, mode='lines'
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([mean + sem, (mean - sem)[::-1]]),
        fill='toself', fillcolor=color, opacity=0.3,
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip", showlegend=False
    ))

def run_session_dynamics(session_id):
    print(f"\n>>> Running Dynamics & Variability Analysis: {session_id}")
    lfp_path = os.path.join(DATA_DIR, f'lfp_by_area_ses-{session_id}.h5')
    if not os.path.exists(lfp_path): return

    with h5py.File(lfp_path, 'r') as hl:
        areas = sorted(list(hl.keys()))
        time_ms = np.linspace(-1000, 5000, 6000)
        
        for area in areas:
            print(f"    Area: {area}")
            
            # Compare conditions: AAAX (Predictable Omission) vs RXRR (Random Omission)
            if 'AAAX' not in hl[area] or 'RXRR' not in hl[area]: continue
            
            # 1. Band-Specific Dynamics (Example: Gamma)
            for band in ['gamma', 'alpha']:
                m_pred, s_pred = compute_trial_tfr_dynamics(hl[f"{area}/AAAX"][()], FS, band)
                m_rand, s_rand = compute_trial_tfr_dynamics(hl[f"{area}/RXRR"][()], FS, band)
                
                fig = go.Figure()
                plot_with_sem(fig, time_ms, m_pred, s_pred, "Predictable (AAAX)", "#CFB87C")
                plot_with_sem(fig, time_ms, m_rand, s_rand, "Random (RXRR)", "cyan")
                
                fig.update_layout(
                    title=f"Session {session_id} - {area} {band.capitalize()} Dynamics",
                    xaxis_title="Time from Stim 1 (ms)", yaxis_title="Power (V^2/Hz)",
                    template="plotly_dark", showlegend=True
                )
                fig.write_html(os.path.join(FIGURES_DIR, f"ses-{session_id}_{area}_{band}_dynamics.html"))

            # 2. Variability Quenching (Variance across trials)
            v_pred = compute_variability_quenching(hl[f"{area}/AAAX"][()])
            v_rand = compute_variability_quenching(hl[f"{area}/RXRR"][()])
            
            fig_v = go.Figure()
            fig_v.add_trace(go.Scatter(x=time_ms, y=v_pred, name="Predictable Var", line=dict(color="#CFB87C")))
            fig_v.add_trace(go.Scatter(x=time_ms, y=v_rand, name="Random Var", line=dict(color="cyan")))
            
            fig_v.update_layout(
                title=f"Session {session_id} - {area} Neuronal Variation (Quenching)",
                xaxis_title="Time from Stim 1 (ms)", yaxis_title="Trial-to-Trial Variance",
                template="plotly_dark"
            )
            fig_v.write_html(os.path.join(FIGURES_DIR, f"ses-{session_id}_{area}_variability.html"))

    gc.collect()

def main():
    lfp_files = [f for f in os.listdir(DATA_DIR) if f.startswith('lfp_by_area') and f.endswith('.h5')]
    for f in lfp_files:
        sid = f.split('_')[-1].split('.')[0].replace('ses-', '')
        run_session_dynamics(sid)

if __name__ == "__main__":
    main()
