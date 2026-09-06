# Agent Memory Bank

Persistent memory and operational handbook for AI agents working with or extending the jnwb library.

## Architectural Philosophy and Invariants

1. **Dataset-Agnostic Library**: jnwb is a generic neuroscience and electrophysiology library. It contains NO dataset-specific condition codes, experiment folder layouts, or paradigm assumptions. Experiment-specific logic belongs in analysis scripts consuming jnwb.
2. **Signal Class Independence**: Spikes (SUA/MUA), local field potentials (LFP), and behavioral signals represent distinct physical observables. Never pool across modalities.
3. **Causal and Directional Language**: Association != Directionality != Causality.
   - Granger causality, phase slope index, and transfer entropy measure temporal-lag asymmetry and predictive information flow.
   - Do not describe directional metrics with physical or interventionist causal verbs.
4. **Logarithm-Last Rule**: When raw power conservation across trials is the estimand, average raw power across trials first, then take the decibel logarithm at the final reporting step.
5. **RNG Reproducibility**: Never mutate global seeds. Always pass an explicit numpy.random.Generator (e.g. rng = np.random.default_rng(seed)).
6. **Public API Contract**: Exactly 105 public symbols are exported in jnwb.__all__. Never break or alter existing public signatures without an approved transition plan.

## NWB Metadata and Addressing Recipe

```python
import jnwb

# 1. Read unit metadata and electrode inventories
units_df = jnwb.get_all_units_metadata("session.nwb")
electrodes_df = jnwb.electrode_inventory("session.nwb")

# 2. Enrich units with standardized area and layer assignments
enriched_units = jnwb.enrich_units_dataframe(units_df, electrodes_df)

# 3. Quality audit
report = jnwb.audit_units(enriched_units)
```

## Spikes, PSTH and Onset Dynamics Recipe

```python
import jnwb
import numpy as np

# Trial-aligned PSTH
time_bins, mean_rate, sem_rate = jnwb.raster_psth(
    spike_times, event_onsets, win_ms=(-200.0, 500.0), bin_ms=10.0
)

# Causal exponential smoothing (no future-leakage)
smooth_rate = jnwb.causal_exp_smooth(mean_rate, bin_ms=10.0, tau_ms=25.0)

# Bounded exponential onset fit
fit = jnwb.fit_exponential_onset(time_bins, smooth_rate, t0_bounds=(0.0, 250.0))
```

## Spectral Analysis and Complex TFR Recipe

```python
import jnwb

# Complex Morlet wavelet transform (returns ComplexTFR with coi_mask)
tfr = jnwb.complex_tfr(lfp_data, fs=1000.0, freqs=freqs)

# Canonical frequency band power
beta_power = jnwb.band_power(
    lfp_data, sampling_rate=1000.0, freq_range=jnwb.CANONICAL_BANDS["beta"]
)
```

## Artifact Detection and Robust Repair Recipe

```python
import jnwb

# 1. Channel correlation matrix & bad channel detection
corr_mat = jnwb.channel_correlation_matrix(lfp_trials)
bad_mask, summary, z_scores = jnwb.bad_channels_from_correlation(corr_mat, z_thresh=5.0)

# 2. Consensus bad trial detection across criteria
bad_trials = jnwb.consensus_bad_trials(lfp_trials, max_amplitude_uv=1500.0)

# 3. Interpolation and trial repair
clean_trials = jnwb.repair_lfp_trials(lfp_trials, bad_mask)
```

## Directed Connectivity and Information Flow Recipe

```python
import jnwb

# Transfer entropy with surrogate nulls
te_result = jnwb.transfer_entropy(x, y, n_surrogates=200, seed=42)

# Phase Slope Index (PSI)
psi_result = jnwb.phase_slope_index(x, y, fs=1000.0, freq_range=(15.0, 30.0))
```

## Statistical Testing and Multiple Comparisons Recipe

```python
import jnwb

# Dual exploratory comparison (parametric + nonparametric)
res = jnwb.StatisticalAnalysis.exploratory_compare(g1, g2)

# Honest Benjamini-Hochberg FDR correction across hypotheses
q_values = jnwb.StatisticalAnalysis.fdr_correct(p_values)
```

## Publication Visualizations and Vector Export Recipe

```python
import jnwb

# Enforce vector text and publication styling
jnwb.setup_vector_graphics()

# Save multi-format figure suite
jnwb.save_figure_suite([fig1, fig2], output_dir="figures/", basename="fig_results")
```

## GPU Acceleration and Parallel Computation Patterns

1. **CuPy / GPU Acceleration**:
   - Modules with GPU support accept `device="cuda"` (with automated fallback to CPU if unavailable).
   - In `jrsa`, CuPy acceleration provides massive speedups for pairwise kernel distance computations: `jrsa(..., backend="cupy")`.
   - In `spectral` and `gpu_pca`, GPU matrix decompositions handle high-density electrode arrays efficiently.
2. **Parallel CPU Execution**:
   - Parallelizable operations support `n_jobs: int = 1` using `joblib.Parallel`.
   - For batch permutation testing or channel-by-channel sweeps, setting `n_jobs=-1` maximizes core utilization.

## Agent Navigation and Skill Tree

When solving electrophysiology tasks, AI agents should consult the canonical skills under `skills/`:
- `skills/jnwb`: Top-level routing kernel and scientific safeguards.
- `skills/jnwb-nwb-data`: NWB I/O, addressing, electrode tables, and unit schemas.
- `skills/jnwb-spiking`: Spike extraction, PSTH, smoothing, and response latencies.
- `skills/jnwb-lfp-spectral`: Wavelets, TFR, PSD, filtering, and artifact repair.
- `skills/jnwb-statistics`: FDR correction, bootstrap CIs, and label permutations.
- `skills/jnwb-population`: Neural trajectories, decoding, and representation ladders.
- `skills/jnwb-connectivity`: Granger causality, phase slope index, and transfer entropy.
- `skills/jnwb-figures`: Publication palettes, auto-scaling axes, and vector SVG exports.
