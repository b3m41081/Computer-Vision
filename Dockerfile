FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        colmap \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY a5/requirements-docker.txt /tmp/requirements.txt
COPY a5/requirements-da3.txt /tmp/requirements-da3.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -m pip install --no-cache-dir -r /tmp/requirements-da3.txt \
    && git clone --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3.git /opt/depth-anything-3 \
    && python -m pip install --no-cache-dir --no-deps --editable /opt/depth-anything-3

ENV DA3_REPO=/opt/depth-anything-3

CMD ["python", "a5/src/extract_video_frames.py"]
