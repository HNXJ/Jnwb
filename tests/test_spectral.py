"""Unit tests for jnwb.spectral -- generic spectral/oscillatory analysis (band power,
cross-area coherence, 1/f tilt, imaginary coherency, re-referencing), promoted 2026-08-23
from omission.jnwb_ext.spectral (99%-jnwb-sufficiency normalization).
"""
from __future__ import annotations

import numpy as np
import pytest

from jnwb.spectral import (
    to_db,
    aggregate_to_db,
    DB_AGGREGATIONS,
    harmonic_analysis,
    cross_area_coherence,
    spectral_tilt,
    band_power,
    imaginary_coherency,
    bipolar_reference,
    laplacian_reference,
    CANONICAL_BANDS,
    compute_psd,
)


class TestPublicImport:
    def test_importable_from_top_level_jnwb(self):
        import jnwb
        assert jnwb.to_db is to_db
        assert jnwb.harmonic_analysis is harmonic_analysis
        assert jnwb.cross_area_coherence is cross_area_coherence
        assert jnwb.spectral_tilt is spectral_tilt
        assert jnwb.band_power is band_power
        assert jnwb.imaginary_coherency is imaginary_coherency
        assert jnwb.bipolar_reference is bipolar_reference
        assert jnwb.laplacian_reference is laplacian_reference
        assert jnwb.CANONICAL_BANDS is CANONICAL_BANDS
        assert jnwb.compute_psd is compute_psd

    def test_listed_in_jnwb_all(self):
        import jnwb
        assert "compute_psd" in jnwb.__all__


class TestComputePsd:
    def test_returns_freqs_and_psd_arrays(self):
        fs = 1000.0
        t = np.arange(0, 2.0, 1.0 / fs)
        x = np.sin(2 * np.pi * 40.0 * t)
        freqs, psd = compute_psd(x, fs)
        assert freqs.shape == psd.shape
        assert freqs[0] == pytest.approx(0.0)

    def test_peak_frequency_recovered(self):
        fs = 1000.0
        t = np.arange(0, 2.0, 1.0 / fs)
        x = np.sin(2 * np.pi * 40.0 * t)
        freqs, psd = compute_psd(x, fs)
        peak = freqs[np.argmax(psd)]
        assert abs(peak - 40.0) < 2.0

    def test_listed_in_jnwb_all(self):
        import jnwb
        for name in ("to_db", "harmonic_analysis", "cross_area_coherence", "spectral_tilt",
                     "band_power", "imaginary_coherency", "bipolar_reference",
                     "laplacian_reference", "CANONICAL_BANDS"):
            assert name in jnwb.__all__


class TestCanonicalBands:
    def test_connectivity_reexports_same_object(self):
        omission_jnwb_ext = pytest.importorskip("omission.jnwb_ext.connectivity")
        assert omission_jnwb_ext.CANONICAL_BANDS is CANONICAL_BANDS

    def test_expected_band_edges(self):
        assert CANONICAL_BANDS["theta"] == (4.0, 8.0)
        assert CANONICAL_BANDS["alpha"] == (8.0, 14.0)
        assert CANONICAL_BANDS["beta"] == (14.0, 30.0)
        assert CANONICAL_BANDS["low_gamma"] == (30.0, 50.0)
        assert CANONICAL_BANDS["high_gamma"] == (50.0, 80.0)


class TestToDb:
    def test_unity_ratio_is_zero_db(self):
        assert to_db(1.0) == pytest.approx(0.0)

    def test_ten_x_ratio_is_ten_db(self):
        assert to_db(10.0) == pytest.approx(10.0)


def _sine(freq_hz, sampling_rate=1000.0, duration_s=2.0, amplitude=1.0, phase=0.0):
    t = np.arange(0, duration_s, 1.0 / sampling_rate)
    return amplitude * np.sin(2 * np.pi * freq_hz * t + phase), t


class TestHarmonicAnalysis:
    def test_empty_input_returns_zeroed_result(self):
        result = harmonic_analysis(np.array([]), sampling_rate=1000.0)
        assert result["fundamental_freq"] == 0.0
        assert result["harmonics"] == {}

    def test_finds_fundamental_frequency_of_pure_tone(self):
        trace, _ = _sine(10.0, sampling_rate=1000.0, duration_s=4.0)
        result = harmonic_analysis(trace, sampling_rate=1000.0, freq_range=(1.0, 90.0))
        assert result["fundamental_freq"] == pytest.approx(10.0, abs=1.0)


class TestCrossAreaCoherence:
    def test_identical_signals_have_high_coherence(self):
        trace, _ = _sine(10.0, sampling_rate=1000.0, duration_s=4.0)
        result = cross_area_coherence(trace, trace, sampling_rate=1000.0)
        assert result["band_coherence"]["theta"] > 0.9

    def test_mismatched_lengths_return_empty_result(self):
        result = cross_area_coherence(np.zeros(100), np.zeros(50), sampling_rate=1000.0)
        assert result["coherence_spectrum"].size == 0

    def test_default_freq_bands_is_canonical_bands(self):
        trace, _ = _sine(10.0, sampling_rate=1000.0, duration_s=4.0)
        result = cross_area_coherence(trace, trace, sampling_rate=1000.0)
        assert set(result["band_coherence"].keys()) == set(CANONICAL_BANDS.keys())


class TestSpectralTilt:
    def test_empty_input_returns_zeroed_result(self):
        result = spectral_tilt(np.array([]), sampling_rate=1000.0)
        assert result["exponent"] == 0.0

    def test_pink_noise_has_negative_exponent(self):
        rng = np.random.default_rng(0)
        white = rng.standard_normal(20000)
        # crude 1/f pink noise via cumulative sum (integrated white noise)
        pink = np.cumsum(white)
        pink -= pink.mean()
        result = spectral_tilt(pink, sampling_rate=1000.0, freq_range=(1.0, 100.0))
        assert result["exponent"] < 0


class TestBandPower:
    def test_empty_input_returns_zero(self):
        assert band_power(np.array([]), sampling_rate=1000.0, freq_range=(4, 8)) == 0.0

    def test_tone_in_band_has_higher_power_than_out_of_band(self):
        trace, _ = _sine(10.0, sampling_rate=1000.0, duration_s=4.0, amplitude=5.0)
        in_band = band_power(trace, 1000.0, (8, 12), normalize=False)
        out_of_band = band_power(trace, 1000.0, (60, 80), normalize=False)
        assert in_band > out_of_band


class TestImaginaryCoherency:
    def test_zero_lag_mixed_source_has_near_zero_icoh(self):
        rng = np.random.default_rng(1)
        source = rng.standard_normal(5000)
        x = source + 0.01 * rng.standard_normal(5000)
        y = source + 0.01 * rng.standard_normal(5000)
        result = imaginary_coherency(x, y, sampling_rate=1000.0, freq_range=(1, 100))
        assert result["coh_mag_mean"] > 0.5
        assert abs(result["icoh_mean"]) < 0.1

    def test_empty_input_returns_zeroed_result(self):
        result = imaginary_coherency(np.array([]), np.array([]), sampling_rate=1000.0, freq_range=(1, 100))
        assert result["n_freqs"] == 0


class TestBipolarReference:
    def test_drops_one_channel(self):
        data = np.arange(12, dtype=float).reshape(4, 3)
        out = bipolar_reference(data)
        assert out.shape == (3, 3)

    def test_common_signal_cancels(self):
        common = np.array([1.0, 2.0, 3.0])
        data = np.stack([common, common, common])
        out = bipolar_reference(data)
        assert np.allclose(out, 0.0)

    def test_rejects_non_2d_input(self):
        with pytest.raises(ValueError):
            bipolar_reference(np.zeros(5))


class TestLaplacianReference:
    def test_preserves_channel_count(self):
        data = np.arange(12, dtype=float).reshape(4, 3)
        out = laplacian_reference(data)
        assert out.shape == (4, 3)

    def test_common_signal_cancels_on_interior_channels(self):
        common = np.array([1.0, 2.0, 3.0])
        data = np.stack([common, common, common, common])
        out = laplacian_reference(data)
        assert np.allclose(out[1:-1], 0.0)


class TestAggregateToDb:
    """The log-last contract: aggregate ratios, take 10*log10 exactly once.

    These tests pin the *defect* the function exists to prevent, not just its happy path --
    averaging decibels is a Jensen error, and the whole point of the primitive is that the
    wrong order is unreachable through it.
    """

    def test_matches_hand_computed_log_last(self):
        power = np.array([[2.0, 4.0]])
        baseline = np.array([[1.0, 2.0]])
        # ratios 2.0 and 2.0 -> mean 2.0 -> 10*log10(2.0)
        got = aggregate_to_db(power, baseline, how="mean_of_ratios", aggregate_over=1)
        np.testing.assert_allclose(got, 10.0 * np.log10(2.0))

    def test_differs_from_averaging_decibels(self):
        """The Jensen gap is real and non-zero for any non-degenerate ratio spread."""
        power = np.array([[1.0, 100.0]])
        baseline = np.ones((1, 2))
        log_last = aggregate_to_db(power, baseline, how="mean_of_ratios", aggregate_over=1)
        log_first = np.mean(to_db(power / baseline), axis=1)
        assert not np.allclose(log_last, log_first)
        # log-last is the larger: mean(log x) <= log(mean x) by Jensen.
        assert log_last[0] > log_first[0]

    def test_two_estimands_actually_differ_when_baseline_varies(self):
        power = np.array([[1.0, 10.0]])
        baseline = np.array([[1.0, 100.0]])
        mor = aggregate_to_db(power, baseline, how="mean_of_ratios", aggregate_over=1)
        rom = aggregate_to_db(power, baseline, how="ratio_of_means", aggregate_over=1)
        assert not np.allclose(mor, rom)

    def test_estimands_coincide_when_baseline_constant(self):
        power = np.array([[2.0, 6.0]])
        baseline = np.full((1, 2), 2.0)
        mor = aggregate_to_db(power, baseline, how="mean_of_ratios", aggregate_over=1)
        rom = aggregate_to_db(power, baseline, how="ratio_of_means", aggregate_over=1)
        np.testing.assert_allclose(mor, rom)

    def test_how_is_required_keyword(self):
        with pytest.raises(TypeError):
            aggregate_to_db(np.ones(3), np.ones(3))

    def test_geometric_mean_rejected_with_reason(self):
        """geomean is mean-of-dB identically, so offering it would ship the bug."""
        r = np.array([1.0, 4.0, 16.0])
        np.testing.assert_allclose(to_db(float(np.exp(np.mean(np.log(r))))), np.mean(to_db(r)))
        with pytest.raises(ValueError, match="deliberately unsupported"):
            aggregate_to_db(np.ones(3), np.ones(3), how="geomean")

    def test_unknown_how_and_nan_policy_rejected(self):
        with pytest.raises(ValueError, match="how must be one of"):
            aggregate_to_db(np.ones(3), np.ones(3), how="median_of_ratios")
        with pytest.raises(ValueError, match="nan_policy"):
            aggregate_to_db(np.ones(3), np.ones(3), how="mean_of_ratios", nan_policy="drop")

    def test_negative_input_raises_as_db_tripwire(self):
        db_like = np.array([-3.0, 1.0, -0.5])
        with pytest.raises(ValueError, match="not ratio-scale power"):
            aggregate_to_db(db_like, np.ones(3), how="mean_of_ratios")
        with pytest.raises(ValueError, match="baseline contains negative"):
            aggregate_to_db(np.ones(3), -np.ones(3), how="mean_of_ratios")

    def test_nan_policy_propagate_vs_omit(self):
        power = np.array([[2.0, np.nan]])
        baseline = np.ones((1, 2))
        assert np.isnan(aggregate_to_db(power, baseline, how="mean_of_ratios",
                                        aggregate_over=1)[0])
        omitted = aggregate_to_db(power, baseline, how="mean_of_ratios",
                                  aggregate_over=1, nan_policy="omit")
        np.testing.assert_allclose(omitted, to_db(2.0))

    def test_no_aggregation_is_elementwise_and_logs_once(self):
        power = np.array([[2.0, 4.0]])
        baseline = np.ones((1, 2))
        np.testing.assert_allclose(
            aggregate_to_db(power, baseline, how="mean_of_ratios"), to_db(power / baseline))

    def test_broadcasts_baseline_against_power(self):
        power = np.array([[2.0, 4.0], [8.0, 16.0]])
        baseline = np.array([2.0, 4.0])
        np.testing.assert_allclose(
            aggregate_to_db(power, baseline, how="mean_of_ratios"), to_db(power / baseline))

    def test_aggregations_constant_is_the_documented_pair(self):
        assert DB_AGGREGATIONS == ("mean_of_ratios", "ratio_of_means")

    def test_importable_from_top_level(self):
        import jnwb
        assert jnwb.aggregate_to_db is aggregate_to_db
        assert jnwb.DB_AGGREGATIONS is DB_AGGREGATIONS
