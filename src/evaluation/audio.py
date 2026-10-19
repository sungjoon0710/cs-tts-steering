"""CPU Praat diagnostics. All derived features require pilot spot-checking."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

SETTINGS = {"version": "praat-diagnostics-v1", "frame_step_s": 0.01,
            "pitch_floor_hz": 60, "pitch_ceiling_hz": 600,
            "silence_threshold_db": -25, "minimum_silent_interval_s": 0.15,
            "minimum_sounding_interval_s": 0.1, "internal_pause_s": 0.2,
            "sensitivity_thresholds_s": [0.15, 0.2, 0.25]}


def extract_audio(path, channel_policy="reject"):
    path = Path(path)
    if not path.exists():
        return {"status": "missing_audio"}
    try:
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as e:
        return {"status": "unreadable_audio", "error_type": type(e).__name__}
    try:
        x, sr = sf.read(path, always_2d=True, dtype="float64")
    except Exception as e:
        return {"status": "unreadable_audio", "audio_sha256": file_hash, "error_type": type(e).__name__}
    result = {"status": "computed_unvalidated", "audio_sha256": file_hash,
              "settings": SETTINGS, "sample_rate": sr, "channels": x.shape[1], "frames": len(x)}
    if not len(x):
        return {**result, "status": "empty_audio"}
    result["duration_s"] = len(x) / sr
    if not np.isfinite(x).all():
        return {**result, "status": "nonfinite_audio"}
    if x.shape[1] != 1 and channel_policy == "reject":
        return {**result, "status": "unsupported_channels", "error": "Explicit mono conversion policy required"}
    if channel_policy not in ("reject", "first", "mean"):
        raise ValueError("Unknown channel policy")
    y = np.mean(x, axis=1) if channel_policy == "mean" else x[:, 0]
    result["channel_policy"] = channel_policy
    rms = float(np.sqrt(np.mean(y * y)))
    result.update(rms_dbfs=float(20 * np.log10(rms)) if rms else None,
                  peak_abs=float(np.max(np.abs(y))), clipped_sample_fraction=float(np.mean(np.abs(y) >= 32767 / 32768)))
    if not rms:
        return {**result, "status": "fully_silent"}
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        return {**result, "status": "pitch_pause_dependency_missing"}
    sound = parselmouth.Sound(y, sampling_frequency=sr)
    try:
        pitch = sound.to_pitch_ac(time_step=0.01, pitch_floor=60, pitch_ceiling=600,
                                  max_number_of_candidates=15, very_accurate=False,
                                  silence_threshold=0.03, voicing_threshold=0.45,
                                  octave_cost=0.01, octave_jump_cost=0.35, voiced_unvoiced_cost=0.14)
        f = pitch.selected_array["frequency"]
        voiced = f[np.isfinite(f) & (f > 0)]
        result["voiced_frame_fraction"] = float(len(voiced) / len(f)) if len(f) else None
        if len(voiced):
            median = float(np.median(voiced))
            st = 12 * np.log2(voiced / median)
            result.update(f0_median_hz=median, f0_std_st=float(np.std(st)),
                          f0_range_st=float(np.percentile(st, 95) - np.percentile(st, 5)),
                          pitch_edge_fraction=float(np.mean((voiced < 66) | (voiced > 540))))
        else:
            result.update(f0_median_hz=None, f0_std_st=None, f0_range_st=None, pitch_status="no_voiced_frames")
        result["f0_track"] = [{"time_s": float(t), "hz": float(v) if v > 0 else None}
                              for t, v in zip(pitch.xs(), f)]
        intensity = sound.to_intensity(minimum_pitch=60, time_step=0.01, subtract_mean=True).values[0]
        result.update(intensity_praat_median_db=float(np.median(intensity)),
                      intensity_praat_std_db=float(np.std(intensity)))
    except Exception as e:
        result.update(pitch_status="extraction_failed", pitch_error_type=type(e).__name__)
    try:
        grid = call(sound, "To TextGrid (silences)", 60, 0.01, -25, 0.15, 0.1, "silent", "sounding")
        intervals = [{"start_s": float(call(grid, "Get start time of interval", 1, i)),
                      "end_s": float(call(grid, "Get end time of interval", 1, i)),
                      "label": call(grid, "Get label of interval", 1, i)}
                     for i in range(1, int(call(grid, "Get number of intervals", 1)) + 1)]
        sounding = [r for r in intervals if r["label"] == "sounding"]
        silence = [r for r in intervals if r["label"] == "silent"]
        result["intensity_intervals"] = intervals
        result["silence_proportion"] = sum(r["end_s"] - r["start_s"] for r in silence) / result["duration_s"]
        result["pause_sensitivity"] = {}
        for threshold in SETTINGS["sensitivity_thresholds_s"]:
            internal = [r["end_s"] - r["start_s"] for r in silence
                        if sounding and r["start_s"] >= sounding[0]["start_s"]
                        and r["end_s"] <= sounding[-1]["end_s"]
                        and r["end_s"] - r["start_s"] + 1e-9 >= threshold]
            result["pause_sensitivity"][str(threshold)] = {"count": len(internal), "total_s": sum(internal),
                "median_s": float(np.median(internal)) if internal else None,
                "p90_s": float(np.percentile(internal, 90)) if internal else None}
        primary = result["pause_sensitivity"]["0.2"]
        result.update(internal_pause_count=primary["count"], internal_pause_total_s=primary["total_s"],
                      internal_pause_proportion=primary["total_s"] / result["duration_s"],
                      articulation_time_estimate_s=(sounding[-1]["end_s"] - sounding[0]["start_s"] - primary["total_s"])
                      if sounding else None, pause_status="computed_unvalidated_intensity_segmentation")
    except Exception as e:
        result.update(pause_status="extraction_failed", pause_error_type=type(e).__name__)
    return result
