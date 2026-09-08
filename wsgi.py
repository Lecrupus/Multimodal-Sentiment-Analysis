"""WSGI entry point for production servers.

    gunicorn  "wsgi:app" -c gunicorn.conf.py      # Linux / macOS / Docker
    waitress-serve --listen=0.0.0.0:8000 wsgi:app  # Windows

Models are loaded once at import so the first request is not penalised. Set
WARMUP=0 to skip that and load them lazily instead.
"""

import logging
import os

from app import app  # noqa: F401  (imported for the WSGI server)
import analysis_logic

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

if os.environ.get("WARMUP", "1") == "1":
    logging.getLogger("wsgi").info("warming up models…")
    analysis_logic.warmup(include_asr=True)
