"""
Scripted initialization latency analysis.

This module computes initialization/transition latencies for different conversation types
using the agent's VAD segments and (for public_phone) template audio snippets in audio_files/.

Inputs:
- combined_timeline: List of segments with keys {start, end, speaker, is_overlap?}
- agent_vad_segments: List[dict] with {start, end}
- processed_wav_path: Path of processed audio (for reference, not required)
- conversation_type: Optional[str] like "public_phone", "private_phone", "web", "presentation", "other" or any custom string

Outputs:
- Dict[str, float] initial_latency_points mapped to stage names in seconds
- Resolved conversation_type string (echo back original or normalized)

Approach:
- For public_phone: match early agent VAD segments to template WAVs attempt2/attempt1/redirect/terminate
  using MFCC DTW similarity. Build a state path and measure time gaps between transitions and the
  next expected segment (e.g., verification, first agent message). If terminate matched, stop.
- For private_phone: measure latency from absolute beginning (0.0) to verification, then to first agent message.
- For web/presentation/other/custom: latency to the first agent VAD segment only (first_agent_message).

Assumptions:
- Agent is right channel; we work only with agent_vad_segments timings from the processed file.
- Verification/disclaimer don't have audio templates; we infer their timing as the next agent VAD
  segment after the last matched template segment (e.g., attempt2/attempt1) for public_phone.

Edge cases covered:
- Missing/empty VAD segments -> return empty dict
- Missing template files -> gracefully skip matching and fall back to generic first message latency
- Sparse segments -> operate on best effort
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import librosa

# =====================
# Configurable settings
# =====================
# Template discovery: will search for these extensions in order
TEMPLATE_EXTS: List[str] = [".mp3", ".wav"]

# Similarity acceptance threshold (lower DTW distance is better). Increase to be less strict.
DTW_ACCEPTANCE_THRESHOLD: float = 0.0150  # tightened per user calibration

# Max number of leading agent segments to try to match against templates
MAX_TEMPLATE_CHECKS: int = 6

# Minimum segment duration (seconds) to attempt template match (avoid tiny clips)
MIN_SEGMENT_SECONDS: float = 0.15

# MFCC/DTW parameters
N_MFCC: int = 20
DTW_METRIC: str = "cosine"

# Verbose: print per-template similarity scores for every checked segment
PRINT_SIMILARITY_SCORES: bool = True


TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "audio_files")


def _resolve_template_path(stem: str) -> Optional[str]:
    """Return an existing path for a template 'stem' using configured extensions."""
    for ext in TEMPLATE_EXTS:
        p = os.path.join(TEMPLATE_DIR, f"{stem}{ext}")
        if os.path.exists(p):
            return p
    return None


def _build_templates() -> Dict[str, Optional[str]]:
    mapping: Dict[str, Optional[str]] = {}
    for name in ("attempt2", "attempt1", "redirect", "terminate"):
        mapping[name] = _resolve_template_path(name)
    return mapping


TEMPLATES = _build_templates()


def _load_audio_mono(path: str, target_sr: int = 16000) -> Optional[Tuple[np.ndarray, int]]:
    if not os.path.exists(path):
        return None
    try:
        y, sr = librosa.load(path, sr=target_sr, mono=True)
        return y, sr
    except Exception:
        return None


def _mfcc_dtw_distance(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    """Compute a normalized DTW distance between two clips via MFCCs.
    Lower is better. Returns +inf on failure.
    """
    try:
        # Pre-emphasis and normalization
        a = librosa.util.normalize(a)
        b = librosa.util.normalize(b)

        mfcc_a = librosa.feature.mfcc(y=a, sr=sr, n_mfcc=N_MFCC)
        mfcc_b = librosa.feature.mfcc(y=b, sr=sr, n_mfcc=N_MFCC)

        # Use cosine distance for DTW
        D, wp = librosa.sequence.dtw(X=mfcc_a, Y=mfcc_b, metric=DTW_METRIC)
        dist = D[-1, -1]
        # Normalize by path length to avoid bias to duration
        norm = len(wp) if len(wp) > 0 else max(mfcc_a.shape[1], mfcc_b.shape[1])
        if norm == 0:
            return float("inf")
        return float(dist / norm)
    except Exception:
        return float("inf")


def _extract_agent_clip(full_audio_path: str, segment: Dict[str, float], target_sr: int = 16000) -> Optional[np.ndarray]:
    """Extract agent-channel audio for given segment timing from processed stereo file.
    Processed file saved by pipeline is stereo at 16k. Right channel is agent.
    """
    try:
        # Load stereo
        y, sr = librosa.load(full_audio_path, sr=target_sr, mono=False)
        if y.ndim == 1:
            # Mono safety: treat as agent
            agent = y
        else:
            # Channels: shape (channels, samples). Right is index 1 when present.
            agent = y[1] if y.shape[0] >= 2 else y[0]
        start = max(0, int(segment["start"] * sr))
        end = max(start + 1, int(segment["end"] * sr))
        return agent[start:end]
    except Exception:
        return None


def _match_segment_to_templates(full_audio_path: str, segment: Dict[str, float]) -> Tuple[Optional[str], float]:
    """Return (template_key, score) for best match. Lower score is better.
    If score > threshold, return (None, score).
    """
    agent_clip = _extract_agent_clip(full_audio_path, segment)
    if agent_clip is None or agent_clip.size < 100:
        return None, float("inf")

    # Load templates lazily and cache in module var
    best_key = None
    best_score = float("inf")
    seg_start = float(segment.get("start", 0.0))
    seg_end = float(segment.get("end", seg_start))
    # Skip extremely short segments
    if (seg_end - seg_start) < MIN_SEGMENT_SECONDS:
        return None, float("inf")

    if PRINT_SIMILARITY_SCORES:
        try:
            print(f"[scripted] Matching segment {seg_start:.2f}-{seg_end:.2f}s against templates (DTW distance; lower is better, threshold={DTW_ACCEPTANCE_THRESHOLD:.4f})...")
        except Exception:
            pass

    for key, path in TEMPLATES.items():
        loaded = _load_audio_mono(path) if path else None
        if not loaded:
            continue
        tpl, sr = loaded
        score = _mfcc_dtw_distance(agent_clip, tpl, sr)
        if PRINT_SIMILARITY_SCORES:
            try:
                sim = 1.0 / (1.0 + score) if np.isfinite(score) else 0.0
                print(f"[scripted]  - {key}: distance={score:.4f} | similarity~{sim:.4f}")
            except Exception:
                pass
        if score < best_score:
            best_score = score
            best_key = key

    # Empirical acceptance: require small distance (lower=better)
    # Accept if distance <= configured threshold
    if PRINT_SIMILARITY_SCORES:
        try:
            if best_key is not None and np.isfinite(best_score):
                sim = 1.0 / (1.0 + best_score)
                print(f"[scripted]  -> best={best_key} with distance={best_score:.4f} (similarity~{sim:.4f})")
            else:
                print(f"[scripted]  -> no acceptable match")
        except Exception:
            pass
    if best_key is not None and best_score <= DTW_ACCEPTANCE_THRESHOLD:
        return best_key, best_score
    return None, best_score


def _first_agent_segment(agent_vad_segments: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not agent_vad_segments:
        return None
    return min(agent_vad_segments, key=lambda s: s.get("start", 0.0))


def _next_agent_segment_after(agent_vad_segments: List[Dict[str, float]], t: float) -> Optional[Dict[str, float]]:
    candidates = [s for s in agent_vad_segments if s.get("start", 0.0) >= t - 1e-6]
    return min(candidates, key=lambda s: s.get("start", 0.0)) if candidates else None


def _merge_segments(segments: List[Dict[str, float]], max_gap: float = 1.5) -> List[Dict[str, float]]:
    """Merge contiguous segments if the gap between them is <= max_gap."""
    if not segments:
        return []
    segments = sorted(
        [{"start": float(s["start"]), "end": float(s["end"])} for s in segments],
        key=lambda s: s["start"],
    )
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        last = merged[-1]
        if seg["start"] - last["end"] <= max_gap:
            last["end"] = max(last["end"], seg["end"])
        else:
            merged.append(seg.copy())
    return merged


def analyze_scripted_initialization(
    combined_timeline: List[Dict[str, float]],
    agent_vad_segments: List[Dict[str, float]],
    processed_wav_path: str,
    conversation_type: Optional[str],
) -> Tuple[Dict[str, float], str]:
    """Main entry. Returns (initial_latency_points, resolved_conversation_type).
    Latencies are in seconds as floats.
    """
    raw_type = (conversation_type or "").strip()
    logic_key = raw_type.lower()
    # Map legacy/variants to new canonical keys
    mapping = {
        "public_phone": "shared-phone",
        "public-phone": "shared-phone",
        "shared_phone": "shared-phone",
        "private_phone": "private-phone",
        "private-phone": "private-phone",
        "web": "web-audio",
        "web-audio": "web-audio",
        "presentation": "presentation",
    }
    resolved_logic = mapping.get(logic_key, logic_key if logic_key in {"shared-phone", "private-phone", "web-audio", "presentation"} else "web-audio")
    # Preserve original string when present, else use resolved_logic
    custom = raw_type if raw_type else resolved_logic

    # Normalize and merge agent segments
    agent_segments = [
        {"start": float(s.get("start", 0.0)), "end": float(s.get("end", 0.0))}
        for s in agent_vad_segments
        if isinstance(s, dict) and "start" in s and "end" in s
    ]
    agent_segments.sort(key=lambda s: s["start"])  # time order
    merged_agent_segments = _merge_segments(agent_segments, max_gap=1.5)

    try:
        print(f"[scripted] type={custom} (logic={resolved_logic}) merged_agent_segments={len(merged_agent_segments)}")
    except Exception:
        pass

    # If nothing to analyze, return empty
    if not merged_agent_segments:
        return {}, custom

    init: Dict[str, float] = {}

    if resolved_logic == "shared-phone":
        # Try to detect template sequence at the beginning via first few agent segments
        # We'll attempt to match up to the first 4-6 agent segments
        max_check = min(MAX_TEMPLATE_CHECKS, len(merged_agent_segments))
        matched: List[Tuple[str, Dict[str, float]]] = []
        for i in range(max_check):
            label, score = _match_segment_to_templates(processed_wav_path, merged_agent_segments[i])
            if label:
                matched.append((label, merged_agent_segments[i]))
            else:
                # Stop matching once a clear non-template appears after we already matched at least one
                if matched:
                    break

        # Determine path transitions and subsequent latencies
        # Valid transitions (based on user prompt):
        # attempt2 -> verification OR attempt1
        # attempt2 -> attempt1 -> verification OR terminate
        # If terminate matched at any point, stop there and record terminate latency
        labels_in_order = [m[0] for m in matched]
        by_label = {m[0]: m[1] for m in matched}

    # We'll compute interval latencies with a moving anchor: initial anchor at t=0, then
        # for agent templates anchor = end of last agent template; for verification anchor = end of last agent template; etc.
        anchor_time = 0.0

        # If redirect matched as the very first, compute disclaimer latency
        if labels_in_order and labels_in_order[0] == "redirect":
            rseg = by_label["redirect"]
            init["disclaimer"] = max(0.0, rseg["start"] - anchor_time)
            anchor_time = rseg["end"]

        if "terminate" in labels_in_order:
            # Compute terminate latency from current anchor
            tseg = by_label["terminate"]
            init["terminate"] = max(0.0, tseg["start"] - anchor_time)
            return init, custom

        first_agent = merged_agent_segments[0]
        # Global simplification: if attempt2 is anywhere, first agent segment is attempt2
        attempt2_anywhere = any(l == "attempt2" for l in labels_in_order)

        if attempt2_anywhere:
            # Force label for first segment for latency reporting
            init["attempt2"] = max(0.0, first_agent["start"] - anchor_time)
            anchor_time = first_agent["end"]

        if "attempt2" in by_label:
            a2 = by_label["attempt2"]
            # Latency before attempt2 from start
            # If we already anchored via first segment above, keep anchor; else use matched a2
            if not attempt2_anywhere:
                init["attempt2"] = max(0.0, a2["start"] - anchor_time)
                anchor_time = a2["end"]
            # Next expected: attempt1 or verification
            after_a2 = _next_agent_segment_after(merged_agent_segments, (first_agent if attempt2_anywhere else a2)["end"] + 1e-6)
            if after_a2 is not None:
                # Try to identify the next segment label
                next_label, _ = _match_segment_to_templates(processed_wav_path, after_a2)
                if next_label == "attempt1":
                    init["attempt1"] = max(0.0, after_a2["start"] - anchor_time)
                    anchor_time = after_a2["end"]
                    # After attempt1: either terminate or verification
                    after_a1 = _next_agent_segment_after(merged_agent_segments, after_a2["end"] + 1e-6)
                    if after_a1 is not None:
                        nl2, _ = _match_segment_to_templates(processed_wav_path, after_a1)
                        if nl2 == "terminate":
                            init["terminate"] = max(0.0, after_a1["start"] - anchor_time)
                        else:
                            # Treat as verification (no template). Latency is from anchor to this start
                            init["verification"] = max(0.0, after_a1["start"] - anchor_time)
                            # After verification, next latency is to first agent message
                            ver_end = after_a1["end"]
                            anchor_time = ver_end
                            # Next agent message after verification
                            after_ver = _next_agent_segment_after(merged_agent_segments, ver_end + 1e-6)
                            if after_ver is not None:
                                init["first_agent_message"] = max(0.0, after_ver["start"] - anchor_time)
                else:
                    # Not attempt1; then treat this as verification (no template)
                    init["verification"] = max(0.0, after_a2["start"] - anchor_time)
                    ver_end = after_a2["end"]
                    anchor_time = ver_end
                    # Next agent message after verification
                    after_ver = _next_agent_segment_after(merged_agent_segments, ver_end + 1e-6)
                    if after_ver is not None:
                        init["first_agent_message"] = max(0.0, after_ver["start"] - anchor_time)
        else:
            # No attempt2 matched; if attempt1 matched, use it similarly
            if "attempt1" in by_label:
                a1 = by_label["attempt1"]
                # Latency before attempt1 from start
                init["attempt1"] = max(0.0, a1["start"] - anchor_time)
                anchor_time = a1["end"]
                after_a1 = _next_agent_segment_after(merged_agent_segments, a1["end"] + 1e-6)
                if after_a1 is not None:
                    nl, _ = _match_segment_to_templates(processed_wav_path, after_a1)
                    if nl == "terminate":
                        init["terminate"] = max(0.0, after_a1["start"] - anchor_time)
                    else:
                        init["verification"] = max(0.0, after_a1["start"] - anchor_time)
                        ver_end = after_a1["end"]
                        anchor_time = ver_end
                        after_ver = _next_agent_segment_after(merged_agent_segments, ver_end + 1e-6)
                        if after_ver is not None:
                            init["first_agent_message"] = max(0.0, after_ver["start"] - anchor_time)

        # If redirect was matched but not first, compute its interval from current anchor
        if "redirect" in by_label and "disclaimer" not in init:
            rseg = by_label["redirect"]
            init["disclaimer"] = max(0.0, rseg["start"] - anchor_time)

        # Required special-case fallbacks per request
        if not labels_in_order:
            # None matched: report first two messages as verification and first agent message
            ver = merged_agent_segments[0]
            init.clear()
            init["verification"] = max(0.0, ver["start"] - 0.0)
            if len(merged_agent_segments) > 1:
                fam = merged_agent_segments[1]
                init["first_agent_message"] = max(0.0, fam["start"] - ver["end"])

        elif ("attempt2" in labels_in_order) and ("attempt1" not in labels_in_order) and ("terminate" in labels_in_order):
            # attempt2 matched, no attempt1, terminate matched: report verification + first agent message only
            ver = merged_agent_segments[0]
            init.clear()
            init["verification"] = max(0.0, ver["start"] - 0.0)
            if len(merged_agent_segments) > 1:
                fam = merged_agent_segments[1]
                init["first_agent_message"] = max(0.0, fam["start"] - ver["end"])

        elif ("attempt2" in labels_in_order) and ("attempt1" in labels_in_order):
            # Ensure we include attempt2, attempt1, verification, first_agent_message (already added above where possible)
            # If verification or first_agent_message missing, try to infer from subsequent segments
            if "verification" not in init:
                # pick the next segment after attempt1 end
                a1seg = by_label["attempt1"]
                after_a1 = _next_agent_segment_after(merged_agent_segments, a1seg["end"] + 1e-6)
                if after_a1 is not None:
                    init["verification"] = max(0.0, after_a1["start"] - max(init.get("attempt1", 0.0) + a1seg["end"] - a1seg["start"], 0.0))
                    next_after_ver = _next_agent_segment_after(merged_agent_segments, after_a1["end"] + 1e-6)
                    if next_after_ver is not None:
                        init["first_agent_message"] = max(0.0, next_after_ver["start"] - after_a1["end"])

        return init, custom

    if resolved_logic == "private-phone":
        # Two latencies: from t=0 to verification, then to first agent message
        # Treat the first merged agent segment as verification
        first = _first_agent_segment(merged_agent_segments)
        if first:
            init["verification"] = max(0.0, first["start"] - 0.0)
            # Next agent after verification is the first agent message
            after_ver = _next_agent_segment_after(merged_agent_segments, first["end"] + 1e-6)
            if after_ver is not None:
                init["first_agent_message"] = max(0.0, after_ver["start"] - first["end"])
        return init, custom

    # web-audio, presentation, other/custom: only first agent message latency
    first = _first_agent_segment(merged_agent_segments)
    if first:
        init["first_agent_message"] = max(0.0, first["start"] - 0.0)
    return init, custom
