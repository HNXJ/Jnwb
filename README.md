# jnwb: Python-Only NWB Analysis & Signal Processing

A dedicated repository for high-performance Python code specialized in Neurodata Without Borders (NWB) files and electrophysiology analysis.

## 📁 Repository Structure

- **`jnwb/`**: The core Python package.
  - `core.py`: Foundational NWB loading and reconstruction utilities.
  - `oglo.py`: Logic for decoding the 'Visual Omission Oddball' paradigm (12 modes).
  - `analysis.py`: Core signal processing (SFC, PEV/Omega-squared).
  - `advanced.py`: (TBD) Advanced laminar and RF mapping logic.
- **`scripts/`**: Ready-to-run analysis pipelines.
  - `prepare_data.py`: Multi-threaded extraction of NWB signals into HDF5 chunks.
  - `run_analysis.py`: Sequential batch processing of experimental sessions.

## 🚀 Key Capabilities

### 1. Paradigm Decoding (OGLO)
Standardized identification of predictable and random omission trials (AAAX, RXRR, etc.) using `jnwb.oglo`.

### 2. Neural Recording Quality
Automatic filtering of "Good Units" based on stability, quality scores, and presence ratios.

### 3. Signal Processing
- **Spike-Field Coherence (SFC)**: Vectorized cross-spectral density estimation between units and local field potentials.
- **Percent Explained Variance (PEV)**: Bias-corrected Omega-squared statistics for neural population variance.

## 🛠️ Installation & Usage

This repository is designed to be used alongside the **Gemini CLI** and the **Office M3 Max** reasoning engine.

```python
import jnwb.oglo as oglo
from jnwb.analysis import compute_spike_field_coherence

# Example: Get trial masks for a session
masks = oglo.get_trial_masks(intervals_df)
```

---
*Maintained by the HNXJ Research Team. Last updated: March 11, 2026.*
