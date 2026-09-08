"""Multimodal sentiment analysis engine.

Every public ``analyze_*`` function returns a plain dictionary so that callers
(the web layer, a CLI, a test) decide how to present it. Nothing here emits
HTML.

The common shape is::

    {
      "ok": bool,
      "modality": "text" | "audio" | "image" | "video",
      "label": "positive" | "neutral" | "negative",
      "confidence": float,          # probability of the winning label
      "valence": float,             # -1.0 .. 1.0
      "distribution": {"positive": float, "neutral": float, "negative": float},
      "detail": {...},              # modality specific evidence
      "warnings": [str],
      "error": str | None,
      "elapsed_ms": int,
    }
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import threading
import time
from typing import Callable, Iterable, Sequence

import numpy as np

from config import Config

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Emotion -> valence mapping
#
# Each raw model label is placed on a -1 (very negative) .. +1 (very positive)
# axis. A label's distance from zero also decides how much probability mass it
# contributes to "neutral", so ambiguous emotions such as surprise stay mostly
# neutral instead of being forced into a polarity.
# --------------------------------------------------------------------------
FACE_VALENCE = {
    "happy": 1.0,
    "surprise": 0.2,
    "neutral": 0.0,
    "sad": -0.75,
    "fear": -0.70,
    "disgust": -0.85,
    "angry": -0.90,
}

AUDIO_VALENCE = {
    "hap": 1.0,
    "neu": 0.0,
    "sad": -0.80,
    "ang": -0.90,
}

TEXT_VALENCE = {
    "positive": 1.0,
    "negative": -1.0,
    "neutral": 0.0,
    # 3-class roberta style checkpoints
    "label_0": -1.0,
    "label_1": 0.0,
    "label_2": 1.0,
}

AUDIO_LABEL_NAMES = {"hap": "happy", "neu": "neutral", "sad": "sad", "ang": "angry"}

SENTIMENTS = ("positive", "neutral", "negative")


class AnalysisError(Exception):
    """Raised for input problems that should be reported to the user."""


# --------------------------------------------------------------------------
# Lazy model loading
#
# Models are heavy, so they are built on first use and cached. The lock keeps
# two concurrent requests from loading the same model twice.
# --------------------------------------------------------------------------
_models: dict[str, object] = {}
_model_lock = threading.Lock()


def _get_model(key: str, factory: Callable[[], object]):
    if key not in _models:
        with _model_lock:
            if key not in _models:
                log.info("loading model: %s", key)
                started = time.time()
                _models[key] = factory()
                log.info("loaded %s in %.1fs", key, time.time() - started)
    return _models[key]


def _text_pipeline():
    def build():
        from transformers import pipeline

        try:
            return pipeline("sentiment-analysis", model=Config.TEXT_MODEL, top_k=None)
        except Exception as exc:
            # Offline, or the download failed. Fall back to the binary model so
            # the app still works, and record that neutral detection is weaker.
            log.warning(
                "could not load %s (%s); falling back to %s",
                Config.TEXT_MODEL,
                exc,
                Config.TEXT_MODEL_FALLBACK,
            )
            pipe = pipeline(
                "sentiment-analysis", model=Config.TEXT_MODEL_FALLBACK, top_k=None
            )
            pipe._is_fallback = True
            return pipe

    return _get_model("text", build)


def _audio_pipeline():
    def build():
        from transformers import pipeline

        return pipeline("audio-classification", model=Config.AUDIO_MODEL, top_k=None)

    return _get_model("audio", build)


def _asr_pipeline():
    def build():
        from transformers import pipeline

        return pipeline(
            "automatic-speech-recognition",
            model=Config.ASR_MODEL,
            chunk_length_s=30,
        )

    return _get_model("asr", build)


def warmup(include_asr: bool = False) -> None:
    """Preload models so the first real request is not slow."""
    _text_pipeline()
    _audio_pipeline()
    if include_asr and Config.ENABLE_ASR:
        _asr_pipeline()


_ffmpeg_path: str | None = None
_ffmpeg_checked = False


def ensure_ffmpeg() -> str | None:
    """Locate ffmpeg, falling back to the copy bundled with imageio-ffmpeg.

    librosa delegates compressed formats to audioread, which searches PATH for
    an ffmpeg executable. Rather than require a system install, the wheel-
    provided binary's directory is prepended to PATH on first use.
    """
    global _ffmpeg_path, _ffmpeg_checked
    if _ffmpeg_checked:
        return _ffmpeg_path

    _ffmpeg_checked = True
    found = shutil.which("ffmpeg")
    if found:
        _ffmpeg_path = found
        return found

    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            folder = os.path.dirname(exe)
            os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
            # audioread looks for a command literally named "ffmpeg"; the wheel
            # ships a versioned filename, so provide an aliased copy once.
            alias = os.path.join(folder, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
            if not os.path.exists(alias):
                try:
                    shutil.copy2(exe, alias)
                except OSError as exc:  # read-only install, e.g. in a container
                    log.warning("could not alias ffmpeg: %s", exc)
            os.environ.setdefault("IMAGEIO_FFMPEG_EXE", exe)
            _ffmpeg_path = alias if os.path.exists(alias) else exe
            log.info("using bundled ffmpeg: %s", _ffmpeg_path)
            return _ffmpeg_path
    except Exception as exc:
        log.warning("bundled ffmpeg unavailable: %s", exc)

    return None


def ffmpeg_available() -> bool:
    return ensure_ffmpeg() is not None


# --------------------------------------------------------------------------
# Distribution helpers
# --------------------------------------------------------------------------
def _to_sentiment(scores: dict[str, float], valence_map: dict[str, float]) -> dict[str, float]:
    """Convert a raw label distribution into positive/neutral/negative.

    A label with valence ``v`` contributes ``max(v, 0)`` to positive,
    ``max(-v, 0)`` to negative and ``1 - abs(v)`` to neutral, so the result is
    still a probability distribution.
    """
    out = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    total = sum(scores.values()) or 1.0
    for label, prob in scores.items():
        p = prob / total
        v = valence_map.get(label.lower(), 0.0)
        out["positive"] += p * max(v, 0.0)
        out["negative"] += p * max(-v, 0.0)
        out["neutral"] += p * (1.0 - abs(v))
    return _normalise(out)


def _normalise(dist: dict[str, float]) -> dict[str, float]:
    total = sum(dist.values())
    if total <= 0:
        return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
    return {k: v / total for k, v in dist.items()}


def _combine(dists: Sequence[dict[str, float]], weights: Sequence[float]) -> dict[str, float]:
    """Weighted average of sentiment distributions."""
    acc = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    total_w = 0.0
    for dist, w in zip(dists, weights):
        if w <= 0:
            continue
        for k in acc:
            acc[k] += dist.get(k, 0.0) * w
        total_w += w
    if total_w <= 0:
        return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
    return _normalise({k: v / total_w for k, v in acc.items()})


def _verdict(dist: dict[str, float]) -> tuple[str, float, float]:
    """Return ``(label, confidence, valence)`` for a sentiment distribution."""
    valence = dist["positive"] - dist["negative"]
    label = max(dist, key=dist.get)
    # A clear winner that is nevertheless very close to the centre is reported
    # as neutral: "slightly positive" is not a useful verdict.
    if label != "neutral" and abs(valence) < Config.NEUTRAL_BAND:
        label = "neutral"
    return label, round(dist[label], 4), round(valence, 4)


def _entropy_confidence(dist: dict[str, float]) -> float:
    """1.0 when the model is decisive, 0.0 when it is maximally unsure."""
    eps = 1e-9
    h = -sum(p * math.log(p + eps) for p in dist.values())
    return max(0.0, min(1.0, 1.0 - h / math.log(len(dist))))


def _result(
    modality: str,
    dist: dict[str, float],
    detail: dict | None = None,
    warnings: list[str] | None = None,
    elapsed_ms: int = 0,
) -> dict:
    label, confidence, valence = _verdict(dist)
    return {
        "ok": True,
        "modality": modality,
        "label": label,
        "confidence": confidence,
        "valence": valence,
        "certainty": round(_entropy_confidence(dist), 4),
        "distribution": {k: round(v, 4) for k, v in dist.items()},
        "detail": detail or {},
        "warnings": warnings or [],
        "error": None,
        "elapsed_ms": elapsed_ms,
    }


def _failure(modality: str, message: str, elapsed_ms: int = 0, detail: dict | None = None) -> dict:
    return {
        "ok": False,
        "modality": modality,
        "label": None,
        "confidence": 0.0,
        "valence": 0.0,
        "certainty": 0.0,
        "distribution": {"positive": 0.0, "neutral": 0.0, "negative": 0.0},
        "detail": detail or {},
        "warnings": [],
        "error": message,
        "elapsed_ms": elapsed_ms,
    }


def _scores_to_dict(raw) -> dict[str, float]:
    """Normalise the several shapes transformers pipelines return."""
    if isinstance(raw, dict):
        raw = [raw]
    if raw and isinstance(raw[0], list):
        raw = raw[0]
    return {item["label"].lower(): float(item["score"]) for item in raw}


# --------------------------------------------------------------------------
# 1. Text
# --------------------------------------------------------------------------
def _window_text(text: str, tokenizer) -> list[str]:
    """Split long text into overlapping windows the model can actually read."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    window = Config.TEXT_WINDOW_TOKENS
    stride = max(1, Config.TEXT_WINDOW_STRIDE)
    if len(ids) <= window:
        return [text]
    chunks = []
    for start in range(0, len(ids), stride):
        piece = ids[start : start + window]
        if not piece:
            break
        chunks.append(tokenizer.decode(piece, skip_special_tokens=True))
        if start + window >= len(ids):
            break
    return chunks


def analyze_text(text: str) -> dict:
    started = time.time()
    warnings: list[str] = []

    if text is None or not text.strip():
        return _failure("text", "Enter some text to analyse.")

    text = text.strip()
    if len(text) > Config.TEXT_MAX_CHARS:
        text = text[: Config.TEXT_MAX_CHARS]
        warnings.append(f"Input truncated to {Config.TEXT_MAX_CHARS:,} characters.")

    try:
        pipe = _text_pipeline()
        if getattr(pipe, "_is_fallback", False):
            warnings.append(
                "Using the binary fallback model: it has no neutral class, so "
                "factual statements may be reported as positive or negative."
            )
        chunks = _window_text(text, pipe.tokenizer)
        if len(chunks) > 1:
            warnings.append(f"Long input analysed as {len(chunks)} overlapping windows.")

        raw_batches = pipe(chunks, truncation=True, max_length=512)
        if isinstance(raw_batches, dict) or (raw_batches and isinstance(raw_batches[0], dict)):
            raw_batches = [raw_batches]

        dists, weights, per_chunk = [], [], []
        for chunk, raw in zip(chunks, raw_batches):
            scores = _scores_to_dict(raw)
            dist = _to_sentiment(scores, TEXT_VALENCE)
            # Longer windows carry proportionally more of the verdict.
            weight = max(1, len(chunk.split()))
            dists.append(dist)
            weights.append(weight)
            label, conf, val = _verdict(dist)
            per_chunk.append(
                {
                    "excerpt": chunk[:160] + ("..." if len(chunk) > 160 else ""),
                    "label": label,
                    "confidence": conf,
                    "valence": val,
                    "words": weight,
                }
            )

        combined = _combine(dists, weights)
        detail = {
            "words": len(text.split()),
            "characters": len(text),
            "windows": len(chunks),
            "per_window": per_chunk if len(per_chunk) > 1 else [],
        }
        return _result("text", combined, detail, warnings, int((time.time() - started) * 1000))

    except Exception as exc:  # pragma: no cover - surfaced to the user
        log.exception("text analysis failed")
        return _failure("text", f"Text analysis failed: {exc}", int((time.time() - started) * 1000))


# --------------------------------------------------------------------------
# 2. Audio
# --------------------------------------------------------------------------
def _sniff_container(path: str) -> str | None:
    """Identify the real container from its magic bytes.

    Extensions lie: WhatsApp exports MPEG-4/AAC audio named ".mp3", which
    libsndfile rejects because it is not MPEG audio at all.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(16)
    except OSError:
        return None

    if head[4:8] == b"ftyp":
        return "MPEG-4 / AAC (M4A)"
    if head[:4] == b"OggS":
        return "Ogg"
    if head[:4] == b"RIFF":
        return "WAV"
    if head[:4] == b"fLaC":
        return "FLAC"
    if head[:3] == b"ID3" or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "MP3"
    return None


def _load_audio(path: str) -> tuple[np.ndarray, float]:
    import librosa

    # Make the bundled ffmpeg discoverable before librosa needs a decoder.
    ensure_ffmpeg()

    try:
        signal, _ = librosa.load(path, sr=Config.AUDIO_SAMPLE_RATE, mono=True)
    except Exception as exc:
        # Several decoder errors arrive with an empty message, which would
        # otherwise surface to the user as "Could not decode the audio ()".
        reason = str(exc).strip() or type(exc).__name__
        container = _sniff_container(path)
        extension = os.path.splitext(path)[1].lstrip(".").upper()

        detail = f"Could not decode the audio ({reason})."
        if container and extension and container.split()[0].upper() != extension:
            detail = (
                f"This file is named .{extension.lower()} but is actually "
                f"{container}. Could not decode it ({reason})."
            )
        if not ffmpeg_available():
            detail += (
                " No ffmpeg backend is available; install ffmpeg or the "
                "imageio-ffmpeg package to support compressed formats."
            )
        raise AnalysisError(detail) from exc

    if signal.size == 0:
        raise AnalysisError("The audio file appears to be empty.")
    return signal, len(signal) / float(Config.AUDIO_SAMPLE_RATE)


def _audio_windows(signal: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """Slice audio into overlapping windows, returning (start_seconds, samples)."""
    sr = Config.AUDIO_SAMPLE_RATE
    size = int(Config.AUDIO_WINDOW_SECONDS * sr)
    hop = max(1, int(Config.AUDIO_HOP_SECONDS * sr))
    if len(signal) <= size:
        return [(0.0, signal)]
    return [
        (start / sr, signal[start : start + size])
        for start in range(0, len(signal) - size + 1, hop)
    ]


def analyze_audio(path: str, transcript_hint: str | None = None) -> dict:
    started = time.time()
    warnings: list[str] = []
    try:
        signal, duration = _load_audio(path)

        if duration > Config.AUDIO_MAX_SECONDS:
            signal = signal[: int(Config.AUDIO_MAX_SECONDS * Config.AUDIO_SAMPLE_RATE)]
            warnings.append(
                f"Only the first {Config.AUDIO_MAX_SECONDS:.0f}s of {duration:.0f}s were analysed."
            )
            duration = Config.AUDIO_MAX_SECONDS

        windows = _audio_windows(signal)
        # Loudness decides how much each window counts: near-silence carries no
        # emotional evidence, but the classifier will still return a label for it.
        energies = np.array([float(np.sqrt(np.mean(w**2)) + 1e-9) for _, w in windows])
        loudest = float(energies.max())
        # The ratio test below is relative to the loudest window, so a file that
        # is silent throughout would otherwise use its own noise floor as the
        # reference and keep everything. Check the absolute level first.
        if loudest < Config.AUDIO_MIN_RMS:
            return _failure(
                "audio",
                "The audio is silent or too quiet to analyse.",
                int((time.time() - started) * 1000),
            )
        keep = energies >= loudest * Config.AUDIO_SILENCE_RATIO
        if not keep.any():
            return _failure("audio", "The audio is silent or too quiet to analyse.")
        if (~keep).any():
            warnings.append(f"Skipped {int((~keep).sum())} near-silent window(s).")

        pipe = _audio_pipeline()
        batch = [w.astype(np.float32) for (_, w), k in zip(windows, keep) if k]
        raw_batches = pipe(batch)
        if raw_batches and isinstance(raw_batches[0], dict):
            raw_batches = [raw_batches]

        dists, weights, timeline = [], [], []
        kept = [(t, e) for (t, _), e, k in zip(windows, energies, keep) if k]
        for (start_s, energy), raw in zip(kept, raw_batches):
            scores = _scores_to_dict(raw)
            dist = _to_sentiment(scores, AUDIO_VALENCE)
            dists.append(dist)
            weights.append(float(energy))
            top = max(scores, key=scores.get)
            label, conf, val = _verdict(dist)
            timeline.append(
                {
                    "t": round(start_s, 2),
                    "emotion": AUDIO_LABEL_NAMES.get(top, top),
                    "label": label,
                    "valence": val,
                    "confidence": conf,
                }
            )

        combined = _combine(dists, weights)

        detail = {
            "duration_seconds": round(duration, 2),
            "windows": len(dists),
            "timeline": timeline,
            "dominant_emotion": _dominant([t["emotion"] for t in timeline]),
        }

        # Optional: transcribe, then blend what was said with how it was said.
        transcript = transcript_hint
        if transcript is None and Config.ENABLE_ASR:
            transcript = _transcribe(path, warnings)
        if transcript:
            text_result = analyze_text(transcript)
            detail["transcript"] = transcript
            if text_result["ok"]:
                detail["transcript_sentiment"] = {
                    "label": text_result["label"],
                    "confidence": text_result["confidence"],
                    "valence": text_result["valence"],
                }
                combined = _combine(
                    [combined, text_result["distribution"]],
                    [Config.FUSION_WEIGHTS["audio"], Config.FUSION_WEIGHTS["text"]],
                )

        return _result("audio", combined, detail, warnings, int((time.time() - started) * 1000))

    except AnalysisError as exc:
        return _failure("audio", str(exc), int((time.time() - started) * 1000))
    except Exception as exc:  # pragma: no cover
        log.exception("audio analysis failed")
        return _failure("audio", f"Audio analysis failed: {exc}", int((time.time() - started) * 1000))


def _transcribe(path: str, warnings: list[str]) -> str | None:
    try:
        out = _asr_pipeline()(path)
        text = (out.get("text") or "").strip() if isinstance(out, dict) else ""
        if not text:
            warnings.append("No speech was detected for transcription.")
            return None
        return text
    except Exception as exc:
        log.warning("transcription failed: %s", exc)
        warnings.append(f"Transcription unavailable ({exc}).")
        return None


def _dominant(labels: Iterable[str]) -> str | None:
    labels = list(labels)
    if not labels:
        return None
    return max(set(labels), key=labels.count)


# --------------------------------------------------------------------------
# 3. Faces (shared by image and video)
# --------------------------------------------------------------------------
def _analyse_faces(frame, enforce: bool, detector: str | None = None) -> list[dict]:
    """Run DeepFace on one image and return a record per *detected* face.

    With ``enforce=False`` DeepFace does not fail when it finds nothing: it
    hands back the whole frame as if it were a face crop, with
    ``face_confidence`` of 0. Feeding an uncropped photo to a model trained on
    tight 48x48 face crops produces confident nonsense, so those records are
    dropped here rather than passed off as a result.
    """
    from deepface import DeepFace

    faces = DeepFace.analyze(
        frame,
        actions=["emotion"],
        detector_backend=detector or Config.FACE_DETECTOR,
        enforce_detection=enforce,
        silent=True,
    )
    if isinstance(faces, dict):
        faces = [faces]

    records = []
    for face in faces:
        emotions = face.get("emotion") or {}
        # DeepFace reports percentages; convert to probabilities.
        scores = {k.lower(): float(v) / 100.0 for k, v in emotions.items()}
        if not scores:
            continue
        region = face.get("region") or {}
        area = int(region.get("w", 0)) * int(region.get("h", 0))
        confidence = float(face.get("face_confidence", 0.0))
        # 0.0 means "nothing was detected here", not "a face I am unsure of".
        if confidence <= Config.FACE_MIN_CONFIDENCE:
            continue
        records.append(
            {
                "scores": scores,
                "dominant": max(scores, key=scores.get),
                "region": {k: int(region.get(k, 0)) for k in ("x", "y", "w", "h")},
                "area": area,
                "face_confidence": round(confidence, 4),
            }
        )
    return records


def _real_faces(records: list[dict]) -> list[dict]:
    """Keep only records that correspond to an actually detected face.

    Guards against the lenient path, where DeepFace returns the entire frame
    with zero confidence and a plausible-looking emotion attached to it.
    """
    return [
        r for r in records
        if r["area"] > 0 and r["face_confidence"] > Config.FACE_MIN_CONFIDENCE
    ]


def _face_records_to_result(records, modality, detail, warnings, started):
    dists, weights, faces_out = [], [], []
    for rec in records:
        dist = _to_sentiment(rec["scores"], FACE_VALENCE)
        label, conf, val = _verdict(dist)
        # Bigger, more confidently detected faces dominate the verdict.
        weights.append(max(1.0, float(rec["area"])) * max(0.1, rec["face_confidence"]))
        dists.append(dist)
        faces_out.append(
            {
                "dominant_emotion": rec["dominant"],
                "label": label,
                "confidence": conf,
                "valence": val,
                "region": rec["region"],
                "face_confidence": rec["face_confidence"],
                "emotions": {k: round(v, 4) for k, v in rec["scores"].items()},
            }
        )
    combined = _combine(dists, weights)
    detail["faces"] = faces_out
    detail["face_count"] = len(faces_out)
    return _result(modality, combined, detail, warnings, int((time.time() - started) * 1000))


def analyze_image(path: str) -> dict:
    started = time.time()
    warnings: list[str] = []
    try:
        import cv2

        image = cv2.imread(path)
        if image is None:
            return _failure("image", "That file could not be read as an image.")

        try:
            records = _analyse_faces(image, enforce=True)
        except Exception:
            # Detection failed. Retry leniently in case a face is present but
            # borderline; genuine misses are filtered out immediately below.
            records = _analyse_faces(image, enforce=False)

        records = _real_faces(records)
        if not records:
            return _failure(
                "image",
                "No face was detected in this image. The detector is trained on "
                "photographs, so drawings, cartoons, avatars and heavily "
                "stylised faces are usually missed, as are very small, blurred, "
                "side-on or partly hidden faces.",
                int((time.time() - started) * 1000),
            )

        h, w = image.shape[:2]
        detail = {"width": int(w), "height": int(h)}
        return _face_records_to_result(records, "image", detail, warnings, started)

    except Exception as exc:  # pragma: no cover
        log.exception("image analysis failed")
        return _failure("image", f"Image analysis failed: {exc}", int((time.time() - started) * 1000))


# --------------------------------------------------------------------------
# 4. Video
# --------------------------------------------------------------------------
def _frame_indices(total_frames: int, fps: float) -> list[int]:
    """Pick frames evenly across the clip, never exceeding the frame budget."""
    if total_frames <= 0:
        return []
    step = max(1, int(round(fps / max(0.01, Config.VIDEO_SAMPLE_FPS))))
    wanted = list(range(0, total_frames, step))
    if len(wanted) > Config.VIDEO_MAX_FRAMES:
        # Spread the budget across the whole video instead of cutting it short.
        idx = np.linspace(0, len(wanted) - 1, Config.VIDEO_MAX_FRAMES)
        wanted = [wanted[int(round(i))] for i in idx]
    return sorted(set(wanted))


def analyze_video(path: str, progress: Callable[[float, str], None] | None = None) -> dict:
    started = time.time()
    warnings: list[str] = []

    def report(pct: float, message: str) -> None:
        if progress:
            progress(max(0.0, min(1.0, pct)), message)

    try:
        import cv2

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return _failure("video", "That file could not be opened as a video.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or math.isnan(fps):
            fps = 25.0
            warnings.append("Frame rate missing from the file; assuming 25 fps.")
        duration = total_frames / fps if total_frames else 0.0

        if duration > Config.VIDEO_MAX_SECONDS:
            total_frames = int(Config.VIDEO_MAX_SECONDS * fps)
            warnings.append(
                f"Only the first {Config.VIDEO_MAX_SECONDS:.0f}s were analysed."
            )
            duration = Config.VIDEO_MAX_SECONDS

        targets = _frame_indices(total_frames, fps)
        if not targets:
            cap.release()
            return _failure("video", "No frames could be read from this video.")

        report(0.02, f"Sampling {len(targets)} frames")

        dists, weights, timeline = [], [], []
        frames_with_faces = 0

        for n, index in enumerate(targets):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            try:
                records = _analyse_faces(frame, enforce=False)
            except Exception:
                continue

            # Frames where nothing was detected must not be counted as faces.
            records = _real_faces(records)
            if not records:
                continue

            frames_with_faces += 1
            best = max(records, key=lambda r: r["area"])
            dist = _to_sentiment(best["scores"], FACE_VALENCE)
            label, conf, val = _verdict(dist)
            dists.append(dist)
            weights.append(1.0)
            timeline.append(
                {
                    "t": round(index / fps, 2),
                    "emotion": best["dominant"],
                    "label": label,
                    "valence": val,
                    "confidence": conf,
                    "faces": len(records),
                }
            )
            if n % 5 == 0:
                report(0.02 + 0.88 * (n / len(targets)), f"Frame {n + 1} of {len(targets)}")

        cap.release()

        if not dists:
            return _failure(
                "video",
                "No faces were detected in any sampled frame.",
                int((time.time() - started) * 1000),
                {"frames_sampled": len(targets), "duration_seconds": round(duration, 2)},
            )

        visual = _combine(dists, weights)
        emotions = [t["emotion"] for t in timeline]
        detail = {
            "duration_seconds": round(duration, 2),
            "fps": round(fps, 2),
            "frames_sampled": len(targets),
            "frames_with_faces": frames_with_faces,
            "face_detection_rate": round(frames_with_faces / len(targets), 3),
            "emotion_counts": {e: emotions.count(e) for e in sorted(set(emotions))},
            "dominant_emotion": _dominant(emotions),
            "timeline": timeline,
            "visual_sentiment": {
                k: round(v, 4) for k, v in visual.items()
            },
        }

        if frames_with_faces < len(targets) * 0.25:
            warnings.append(
                f"A face was found in only {frames_with_faces} of {len(targets)} sampled frames."
            )

        combined = visual

        # Blend the soundtrack in when ffmpeg can hand us the audio track.
        report(0.92, "Analysing audio track")
        audio_result = _video_audio(path, warnings)
        if audio_result and audio_result["ok"]:
            detail["audio"] = {
                "label": audio_result["label"],
                "confidence": audio_result["confidence"],
                "valence": audio_result["valence"],
                "dominant_emotion": audio_result["detail"].get("dominant_emotion"),
                "transcript": audio_result["detail"].get("transcript"),
            }
            combined = _combine(
                [visual, audio_result["distribution"]],
                [Config.FUSION_WEIGHTS["face"], Config.FUSION_WEIGHTS["audio"]],
            )
            detail["fusion"] = "face + audio"
        else:
            detail["fusion"] = "face only"

        report(1.0, "Done")
        return _result("video", combined, detail, warnings, int((time.time() - started) * 1000))

    except Exception as exc:  # pragma: no cover
        log.exception("video analysis failed")
        return _failure("video", f"Video analysis failed: {exc}", int((time.time() - started) * 1000))


def _video_audio(path: str, warnings: list[str]) -> dict | None:
    """Analyse a video's audio track, if the toolchain allows it."""
    if not ffmpeg_available():
        warnings.append(
            "ffmpeg was not found, so the audio track was skipped and the "
            "verdict is based on facial expressions alone."
        )
        return None
    try:
        return analyze_audio(path)
    except Exception as exc:
        log.warning("video audio analysis failed: %s", exc)
        warnings.append(f"Audio track could not be analysed ({exc}).")
        return None


# --------------------------------------------------------------------------
# 5. Explicit multimodal fusion
# --------------------------------------------------------------------------
def fuse(results: dict[str, dict]) -> dict:
    """Combine several modality results into one overall verdict.

    ``results`` maps a modality name to a result dict; failed or missing
    modalities are ignored.
    """
    dists, weights, used = [], [], []
    for modality, result in results.items():
        if not result or not result.get("ok"):
            continue
        weight_key = "face" if modality in {"image", "video"} else modality
        weight = Config.FUSION_WEIGHTS.get(weight_key, 1.0)
        # Trust a decisive modality more than a hesitant one.
        weight *= 0.5 + 0.5 * float(result.get("certainty", 0.5))
        dists.append(result["distribution"])
        weights.append(weight)
        used.append(modality)

    if not dists:
        return _failure("fusion", "No modality produced a usable result.")

    combined = _combine(dists, weights)
    agreement = len({results[m]["label"] for m in used}) == 1
    return _result(
        "fusion",
        combined,
        {
            "modalities": used,
            "agreement": agreement,
            "per_modality": {
                m: {
                    "label": results[m]["label"],
                    "confidence": results[m]["confidence"],
                    "valence": results[m]["valence"],
                }
                for m in used
            },
        },
        [] if agreement or len(used) < 2 else ["Modalities disagree; treat the verdict with care."],
    )
