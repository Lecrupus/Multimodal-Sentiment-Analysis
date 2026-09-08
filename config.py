"""Central configuration.

Every value can be overridden with an environment variable of the same name,
so deployments do not need to edit code.
"""

import os


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-key-change-me-in-production")
    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploads")

    # Reject oversized uploads before they are written to disk.
    MAX_CONTENT_LENGTH = _env_int("MAX_UPLOAD_MB", 100) * 1024 * 1024

    ALLOWED_IMAGE = {"png", "jpg", "jpeg", "bmp", "webp"}
    ALLOWED_AUDIO = {"wav", "mp3", "flac", "ogg", "m4a"}
    ALLOWED_VIDEO = {"mp4", "mov", "avi", "mkv", "webm"}

    # --- Text ---
    # A 3-class model (negative/neutral/positive). Binary SST-2 checkpoints
    # have no neutral class and label factual sentences with high confidence,
    # so this is the default; TEXT_MODEL_FALLBACK is used if it cannot load.
    TEXT_MODEL = os.environ.get(
        "TEXT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    TEXT_MODEL_FALLBACK = os.environ.get(
        "TEXT_MODEL_FALLBACK",
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    )
    # Longer inputs are split into overlapping windows and aggregated.
    TEXT_WINDOW_TOKENS = _env_int("TEXT_WINDOW_TOKENS", 400)
    TEXT_WINDOW_STRIDE = _env_int("TEXT_WINDOW_STRIDE", 350)
    TEXT_MAX_CHARS = _env_int("TEXT_MAX_CHARS", 50_000)

    # --- Audio ---
    AUDIO_MODEL = os.environ.get("AUDIO_MODEL", "superb/wav2vec2-base-superb-er")
    AUDIO_SAMPLE_RATE = 16_000
    # The model was trained on short utterances, so long files are windowed.
    AUDIO_WINDOW_SECONDS = _env_float("AUDIO_WINDOW_SECONDS", 4.0)
    AUDIO_HOP_SECONDS = _env_float("AUDIO_HOP_SECONDS", 2.0)
    AUDIO_MAX_SECONDS = _env_float("AUDIO_MAX_SECONDS", 600.0)
    # Windows quieter than this (relative to the loudest window) are ignored.
    AUDIO_SILENCE_RATIO = _env_float("AUDIO_SILENCE_RATIO", 0.08)
    # Absolute RMS floor. The ratio above is relative, so without this a
    # completely silent file would treat its own silence as the reference
    # level and pass every window through.
    AUDIO_MIN_RMS = _env_float("AUDIO_MIN_RMS", 1e-4)

    # --- Vision ---
    FACE_DETECTOR = os.environ.get("FACE_DETECTOR", "opencv")
    # DeepFace reports face_confidence 0.0 when it detected nothing and simply
    # handed back the whole frame. Anything at or below this is not a face, so
    # its "emotion" is meaningless and must not reach a verdict.
    FACE_MIN_CONFIDENCE = _env_float("FACE_MIN_CONFIDENCE", 0.0)

    # --- Video ---
    VIDEO_SAMPLE_FPS = _env_float("VIDEO_SAMPLE_FPS", 1.0)
    # Hard ceiling on analysed frames; frames are spread evenly across the clip
    # so a long video is summarised rather than truncated.
    VIDEO_MAX_FRAMES = _env_int("VIDEO_MAX_FRAMES", 120)
    VIDEO_MAX_SECONDS = _env_float("VIDEO_MAX_SECONDS", 900.0)

    # --- Optional speech-to-text (adds transcript sentiment to audio/video) ---
    # Off by default: enabling downloads an extra model on first use.
    ENABLE_ASR = _env_bool("ENABLE_ASR", False)
    ASR_MODEL = os.environ.get("ASR_MODEL", "openai/whisper-tiny")

    # --- Fusion weights (relative importance per modality) ---
    FUSION_WEIGHTS = {
        "text": _env_float("WEIGHT_TEXT", 1.0),
        "audio": _env_float("WEIGHT_AUDIO", 0.8),
        "face": _env_float("WEIGHT_FACE", 0.9),
    }

    # Below this absolute valence the verdict is reported as neutral.
    NEUTRAL_BAND = _env_float("NEUTRAL_BAND", 0.15)

    # Finished job results are kept in memory this long.
    JOB_TTL_SECONDS = _env_int("JOB_TTL_SECONDS", 3600)
