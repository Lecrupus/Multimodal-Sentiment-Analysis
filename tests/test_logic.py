"""Tests for the analysis engine.

The pure maths (distributions, fusion, weighting) is tested without loading any
model. Model-backed tests are marked slow and skipped unless RUN_SLOW=1.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analysis_logic as A  # noqa: E402
from config import Config  # noqa: E402

slow = pytest.mark.skipif(
    os.environ.get("RUN_SLOW") != "1", reason="set RUN_SLOW=1 to run model tests"
)


# --------------------------------------------------------------- distributions
def test_sentiment_mapping_is_a_distribution():
    dist = A._to_sentiment({"happy": 0.5, "sad": 0.3, "neutral": 0.2}, A.FACE_VALENCE)
    assert pytest.approx(sum(dist.values()), abs=1e-6) == 1.0
    assert set(dist) == {"positive", "neutral", "negative"}


def test_pure_happy_is_positive():
    dist = A._to_sentiment({"happy": 1.0}, A.FACE_VALENCE)
    assert dist["positive"] == pytest.approx(1.0)


def test_ambiguous_emotion_lands_mostly_neutral():
    # Surprise has a small valence, so most of its mass must stay neutral.
    dist = A._to_sentiment({"surprise": 1.0}, A.FACE_VALENCE)
    assert dist["neutral"] > dist["positive"]


def test_neutral_label_is_neutral():
    dist = A._to_sentiment({"neutral": 1.0}, A.FACE_VALENCE)
    assert dist["neutral"] == pytest.approx(1.0)


def test_empty_scores_do_not_crash():
    assert A._normalise({"positive": 0.0, "neutral": 0.0, "negative": 0.0})["neutral"] == 1.0


# --------------------------------------------------------------- verdicts
def test_verdict_picks_the_winner():
    label, conf, val = A._verdict({"positive": 0.7, "neutral": 0.2, "negative": 0.1})
    assert label == "positive"
    assert conf == pytest.approx(0.7)
    assert val == pytest.approx(0.6)


def test_marginal_polarity_reports_as_neutral():
    # Winner is positive but valence sits inside the neutral band.
    label, _, _ = A._verdict({"positive": 0.36, "neutral": 0.34, "negative": 0.30})
    assert label == "neutral"


def test_certainty_is_high_when_decisive():
    decisive = A._entropy_confidence({"positive": 0.98, "neutral": 0.01, "negative": 0.01})
    unsure = A._entropy_confidence({"positive": 0.34, "neutral": 0.33, "negative": 0.33})
    assert decisive > 0.8
    assert unsure < 0.05


# --------------------------------------------------------------- combining
def test_combine_respects_weights():
    a = {"positive": 1.0, "neutral": 0.0, "negative": 0.0}
    b = {"positive": 0.0, "neutral": 0.0, "negative": 1.0}
    assert A._combine([a, b], [1, 1])["positive"] == pytest.approx(0.5)
    assert A._combine([a, b], [3, 1])["positive"] == pytest.approx(0.75)


def test_zero_weights_are_ignored():
    a = {"positive": 1.0, "neutral": 0.0, "negative": 0.0}
    b = {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
    assert A._combine([a, b], [1, 0])["positive"] == pytest.approx(1.0)


# --------------------------------------------------------------- windowing
def test_audio_windows_cover_long_signals():
    sr = Config.AUDIO_SAMPLE_RATE
    signal = np.zeros(sr * 10, dtype=np.float32)
    windows = A._audio_windows(signal)
    assert len(windows) > 1
    assert windows[0][0] == 0.0
    assert all(len(w) == int(Config.AUDIO_WINDOW_SECONDS * sr) for _, w in windows)


def test_short_audio_is_one_window():
    signal = np.zeros(Config.AUDIO_SAMPLE_RATE, dtype=np.float32)
    assert len(A._audio_windows(signal)) == 1


def test_frame_budget_is_never_exceeded():
    idx = A._frame_indices(total_frames=200_000, fps=30.0)
    assert len(idx) <= Config.VIDEO_MAX_FRAMES
    assert idx == sorted(idx)


def test_frame_sampling_spans_the_whole_clip():
    idx = A._frame_indices(total_frames=100_000, fps=30.0)
    # The last sampled frame should be near the end, not near the start.
    assert idx[-1] > 90_000


def test_no_frames_for_empty_video():
    assert A._frame_indices(0, 30.0) == []


# --------------------------------------------------------------- fusion
def _ok(modality, dist, certainty=0.9):
    return {
        "ok": True, "modality": modality, "label": max(dist, key=dist.get),
        "confidence": max(dist.values()), "valence": dist["positive"] - dist["negative"],
        "certainty": certainty, "distribution": dist, "detail": {},
        "warnings": [], "error": None, "elapsed_ms": 1,
    }


def test_fusion_notes_agreement():
    pos = {"positive": 0.9, "neutral": 0.1, "negative": 0.0}
    fused = A.fuse({"text": _ok("text", pos), "image": _ok("image", pos)})
    assert fused["ok"]
    assert fused["label"] == "positive"
    assert fused["detail"]["agreement"] is True
    assert not fused["warnings"]


def test_fusion_warns_on_disagreement():
    pos = {"positive": 0.9, "neutral": 0.1, "negative": 0.0}
    neg = {"positive": 0.0, "neutral": 0.1, "negative": 0.9}
    fused = A.fuse({"text": _ok("text", pos), "image": _ok("image", neg)})
    assert fused["detail"]["agreement"] is False
    assert fused["warnings"]


def test_fusion_ignores_failed_modalities():
    pos = {"positive": 0.9, "neutral": 0.1, "negative": 0.0}
    fused = A.fuse({
        "text": _ok("text", pos),
        "audio": {"ok": False, "error": "boom", "distribution": {}},
    })
    assert fused["detail"]["modalities"] == ["text"]


def test_fusion_with_nothing_usable_fails_cleanly():
    fused = A.fuse({"text": {"ok": False, "error": "x"}})
    assert fused["ok"] is False
    assert fused["error"]


def test_decisive_modality_outweighs_hesitant_one():
    pos = {"positive": 0.9, "neutral": 0.1, "negative": 0.0}
    neg = {"positive": 0.0, "neutral": 0.1, "negative": 0.9}
    fused = A.fuse({
        "text": _ok("text", pos, certainty=1.0),
        "image": _ok("image", neg, certainty=0.0),
    })
    assert fused["valence"] > 0


# --------------------------------------------------------------- error paths
def test_blank_text_is_rejected_without_a_model():
    result = A.analyze_text("   ")
    assert result["ok"] is False
    assert "Enter some text" in result["error"]


def test_missing_image_file_fails_cleanly():
    result = A.analyze_image("does-not-exist.jpg")
    assert result["ok"] is False
    assert result["error"]


def test_missing_video_file_fails_cleanly():
    result = A.analyze_video("does-not-exist.mp4")
    assert result["ok"] is False


def test_failure_shape_matches_success_shape():
    ok = A._result("text", {"positive": 1.0, "neutral": 0.0, "negative": 0.0})
    bad = A._failure("text", "nope")
    assert set(ok) == set(bad)


# --------------------------------------------------------------- model-backed
@slow
def test_text_polarity_end_to_end():
    assert A.analyze_text("I love this, it is wonderful!")["label"] == "positive"
    assert A.analyze_text("Awful, broken and a waste of money.")["label"] == "negative"


@slow
def test_long_text_is_windowed():
    result = A.analyze_text("This film was fantastic and moving. " * 200)
    assert result["ok"]
    assert result["detail"]["windows"] > 1


@slow
def test_silent_audio_is_reported(tmp_path):
    import soundfile as sf

    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(16000 * 3, dtype=np.float32), 16000)
    result = A.analyze_audio(str(path))
    assert result["ok"] is False
    assert "silent" in result["error"].lower()


# --------------------------------------------------------------- face filtering
def _rec(confidence, area=10_000, dominant="happy"):
    return {
        "scores": {dominant: 0.9, "neutral": 0.1},
        "dominant": dominant,
        "region": {"x": 0, "y": 0, "w": 100, "h": 100},
        "area": area,
        "face_confidence": confidence,
    }


def test_zero_confidence_face_is_dropped():
    # DeepFace returns the whole frame with confidence 0 when it detects
    # nothing; that is not a face and must never reach a verdict.
    assert A._real_faces([_rec(0.0)]) == []


def test_detected_face_is_kept():
    assert len(A._real_faces([_rec(0.93)])) == 1


def test_zero_area_is_dropped():
    assert A._real_faces([_rec(0.9, area=0)]) == []


def test_mixed_records_keep_only_real_faces():
    kept = A._real_faces([_rec(0.0), _rec(0.88), _rec(0.0), _rec(0.42)])
    assert [r["face_confidence"] for r in kept] == [0.88, 0.42]


def test_face_weighting_uses_confidence_not_falsy_default():
    # A confidence of 0.0 must not be silently promoted to full weight.
    detail = {}
    result = A._face_records_to_result(
        [_rec(0.9, area=40_000, dominant="happy"), _rec(0.9, area=100, dominant="angry")],
        "image", detail, [], __import__("time").time(),
    )
    # The much larger happy face should dominate the much smaller angry one.
    assert result["label"] == "positive"
