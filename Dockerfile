FROM python:3.12.16-slim-bookworm@sha256:623674a14ef93d052bb9b0de4dbf5f28eb578a35263838941cc8aa43213c1487

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /processor

# PyTorch build: cpu or cuda. app.yml's variants set it (cpu variant: CPU-only torch from the PyTorch CPU index,
# no NVIDIA libraries; gpu variant: the CUDA 12.6 build for the platform's driver 550, which also runs on CPU
# with USE_GPU=auto). The default is cpu for a builder that passes no build arguments.
ARG TORCH_FLAVOUR=cpu

# The complete pinned dependency set (no resolver: see requirements.txt).
COPY requirements.txt requirements-${TORCH_FLAVOUR}.txt /processor/
RUN pip install --no-deps --prefer-binary \
        -r /processor/requirements.txt -r /processor/requirements-${TORCH_FLAVOUR}.txt \
    && find /usr/local/lib/python3.12/site-packages -name '*.pyc' -delete \
    && find /usr/local/lib/python3.12/site-packages -path '*/torch/*' -prune -o -type d -name tests -prune -exec rm -rf {} + \
    && rm -rf /usr/local/lib/python3.12/site-packages/torch/include

# SuperAnimal weights (Mathis lab; academic, non-commercial use only: licenses/SuperAnimal-LICENSE.txt), downloaded
# from Hugging Face at pinned revisions and checked against their SHA-256 (processor/weights.py), into the folder
# DeepLabCut reads SuperAnimal snapshots from. Nothing is downloaded at run time.
COPY processor/__init__.py processor/weights.py /processor/processor/
RUN HF_HOME=/tmp/hf python -m processor.weights download && rm -rf /tmp/hf

COPY processor/ /processor/processor/
COPY VERSION /processor/VERSION
COPY LICENSE THIRD_PARTY.md /processor/
COPY licenses/ /processor/licenses/

ENV PYTHONPATH="/processor" \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba \
    XDG_CACHE_HOME=/tmp/cache \
    TORCH_HOME=/tmp/torch \
    HF_HOME=/tmp/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TORCH_FLAVOUR=${TORCH_FLAVOUR}

# Build matplotlib's font cache and import the stack once, so a run needs no writes outside /tmp and no network.
RUN python -c "import matplotlib.pyplot, deeplabcut, ndx_pose, pynwb, cv2, imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" \
    && python -m processor.weights verify \
    && rm -rf /tmp/* && mkdir -p /tmp/matplotlib /tmp/numba /tmp/cache && chmod -R a+rwX /tmp

# Runs as uid 1000 on the platform; nothing here needs root. PyTorch looks the user up (getpass), so uid 1000
# gets a passwd entry.
RUN useradd --uid 1000 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin processor
USER 1000

ENTRYPOINT ["python3", "-m", "processor.entry"]
