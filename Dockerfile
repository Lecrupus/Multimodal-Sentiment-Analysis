# syntax=docker/dockerfile:1
#
# Works on Hugging Face Spaces (sdk: docker), and on any container host.
# Spaces expects the app on port 7860 and runs the container as UID 1000.

FROM python:3.11-slim

# libGL/libglib are needed by OpenCV. ffmpeg also arrives via the
# imageio-ffmpeg wheel, but the system build is faster and smaller to load.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces runs as UID 1000; everything the app writes must live under its home.
RUN useradd --create-home --uid 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface \
    DEEPFACE_HOME=/home/user \
    MPLCONFIGDIR=/tmp/mpl \
    PORT=7860

USER user
WORKDIR /home/user/app

# Install dependencies first so code edits do not invalidate this layer.
COPY --chown=user requirements.txt .
RUN pip install --user --timeout 180 --retries 10 -r requirements.txt

COPY --chown=user . .

# Bake the model weights into the image. Without this, the first request on a
# cold Space waits several minutes on ~700 MB of downloads. Set to 0 to skip
# and fetch them lazily at runtime instead.
ARG PREFETCH_MODELS=1
RUN if [ "$PREFETCH_MODELS" = "1" ]; then \
      python -c "import analysis_logic; analysis_logic.warmup()" && \
      python -c "import numpy as np; from deepface import DeepFace; \
DeepFace.analyze(np.zeros((224,224,3), dtype='uint8'), actions=['emotion'], enforce_detection=False)" ; \
    fi

RUN mkdir -p /home/user/app/uploads

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:7860/api/health').read()"

CMD ["gunicorn", "wsgi:app", "-c", "gunicorn.conf.py"]
