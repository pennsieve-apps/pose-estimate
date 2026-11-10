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
    && find /usr/local/lib/python3.12/site-packages -path '*/torch/*' -prune -o -path '*/tables/*' -prune -o -type d -name tests -prune -exec rm -rf {} + \
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

# Lambda-safe: the weights live in the read-only image (site-packages); every writable cache (matplotlib,
# numba, torch, Hugging Face) is pointed at $TMPDIR at start-up by processor/entry.py, nothing relies on the
# image's /tmp or HOME, and the Hugging Face hub is offline. USER is set for getpass (any uid works).
ENV PYTHONPATH="/processor" \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    HF_HUB_OFFLINE=1 \
    USER=app \
    TORCH_FLAVOUR=${TORCH_FLAVOUR}

# Import the stack once and check the weights (as root at build time; nothing is kept from this in /tmp).
RUN python -c "import matplotlib.pyplot, deeplabcut, ndx_pose, pynwb, cv2" \
    && python -m processor.weights verify \
    && rm -rf /tmp/* /root/.cache

# Runs as uid 1000 on ECS and as an arbitrary least-privileged user on Lambda; nothing here needs root.
RUN useradd --uid 1000 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin app
USER 1000

ENTRYPOINT ["python3", "-m", "processor.entry"]
