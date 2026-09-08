"""Gunicorn settings tuned for a CPU inference workload.

The models are large and thread-safe enough for this use, so a small number of
workers each serving several threads uses far less memory than many processes.
"""

import multiprocessing
import os

# Hugging Face Spaces routes to 7860; override with PORT elsewhere.
bind = f"0.0.0.0:{os.environ.get('PORT', '7860')}"

# Each worker loads its own copy of the models, so keep the count low.
workers = int(os.environ.get("WEB_CONCURRENCY", 1))
threads = int(os.environ.get("WEB_THREADS", 4))
worker_class = "gthread"

# Video analysis is slow by nature; do not let the arbiter kill a live job.
timeout = int(os.environ.get("WEB_TIMEOUT", 1800))
graceful_timeout = 60
keepalive = 5

# Load the app before forking so workers share model memory copy-on-write.
preload_app = os.environ.get("PRELOAD", "1") == "1"

accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info").lower()

max_requests = int(os.environ.get("MAX_REQUESTS", 200))
max_requests_jitter = 20
