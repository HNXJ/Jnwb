---
name: jnwb
description: Top-level router, scientific safeguard kernel, and memory bank for jnwb NWB electrophysiology analysis.
---

# jnwb — Neuroscience & Electrophysiology Analysis Kernel

## 1. Trigger
Activate this skill when the user asks for generic electrophysiology analysis, time-frequency analysis, spike dynamics, NWB processing, neural statistics, decoding, artifact rejection, or directed connectivity.

## 2. Task-to-Primitive Routing Matrix
- **NWB inspection, paths, metadata, electrodes, addressing, compression**: delegate to `jnwb-nwb-data`
- **Spike raster/PSTH, latency estimation, causal smoothing, unit QC**: delegate to `jnwb-spiking`
- **LFP filtering, complex Morlet TFR, multi-trial accumulation, artifact repair**: delegate to `jnwb-lfp-spectral`
- **Bootstrap, label/trial permutation, multiple comparisons (FDR), RNG safety**: delegate to `jnwb-statistics`
- **Linear SVM decoding, neural trajectories, jRSA, population geometry**: delegate to `jnwb-population`
- **Directional coupling (Granger, PSI, transfer entropy) with strict causal language**: delegate to `jnwb-connectivity`
- **Visual QC, raster PSTH plotting, multi-format figure export**: delegate to `jnwb-figures`

## 3. High-Performance Acceleration (CuPy & Joblib)
- **GPU / CuPy Acceleration**: Operations supporting GPU execution accept `device='cuda'` (with automated fallback to CPU if unavailable). Use `backend='cupy'` for distance matrix speedups in `jrsa`.
- **Parallel CPU Processing**: Batch shuffles, permutation testing, and pairwise channel matrices support `n_jobs: int = 1` (or `n_jobs=-1` for all CPU cores) using `joblib.Parallel`.
- **Artifact Rejection & Repair**: Pre-filter LFP matrices using `bad_channels_from_correlation`, `consensus_bad_trials`, and `repair_lfp_trials`.

## 4. Core Scientific Safeguards & Invariants
1. **Signal Class Independence**: Spikes (SUA/MUA) and continuous LFP represent distinct physical observables. Never pool across modalities.
2. **Estimand & Causal Hierarchy**: $\text{Association} \ne \text{Directionality} \ne \text{Causality}$. Granger causality and phase slope index measure temporal-lag asymmetry (predictive directionality), not anatomical/physical causality.
3. **Logarithm Last**: For spectral power or decibel changes: average raw power across trials first, normalize by baseline, and compute $10 \cdot \log_{10}(\text{power})$ at the final step.
4. **Boundary & Filter Distortions**: Mask wavelet coefficients in the Cone of Influence (`coi_mask`). Use causal exponential smoothing (`causal_exp_smooth`) to prevent future leakage.
5. **RNG Reproducibility**: Pass explicit `numpy.random.Generator` instances (e.g. `rng = np.random.default_rng(seed)`). Never mutate global `np.random.seed()`.
6. **Dataset-Agnostic Invariant**: `jnwb` is dataset-agnostic. Experiment-specific condition codes and folder layouts belong in user analysis scripts, never in `jnwb`.

## 5. Agent Memory & Operational Guidance
For detailed workflow recipes, memory conventions, and common AI agent pitfalls, see:
- [AGENTS.md](../../AGENTS.md) — Authoritative repository operational contract and PRGS execution grammar.
- [docs/memory.md](../../docs/memory.md) — Comprehensive agent memory bank for end-to-end NWB workflows.

## 6. Minimal Workflow
```python
import jnwb
import numpy as np

rng = np.random.default_rng(42)
data = rng.normal(size=(500,))
freqs = np.array([10.0, 20.0, 40.0])
tfr = jnwb.complex_tfr(data, fs=1000.0, freqs=freqs)
```

## 7. Verification
- All 101 exports resolve from `import jnwb`.
- `mkdocs build --strict` and `sphinx-build -W` compile docs warning-free.
- `pytest tests/` passes 446+ tests.

