"""
Container entry point: points every writable cache at $TMPDIR (the only writable place besides
HOME, INPUT_DIR and OUTPUT_DIR on the platform's Lambda runtime; nothing baked into the image's
/tmp or HOME survives there), fixes the thread count from THREADS (or the cgroup CPU quota, never
the host's core count) before numpy or torch is imported, then runs main() once (pose-estimate).
"""
import os
import sys
import tempfile

from processor.resources import resolve_threads, set_thread_env


def set_cache_env() -> None:
    for var in ("TMPDIR", "HOME"):       # the runtime creates them; a local test may not have
        if os.environ.get(var):
            try:
                os.makedirs(os.environ[var], exist_ok=True)
            except OSError:
                pass
    tempfile.tempdir = None
    tmp = tempfile.gettempdir()           # $TMPDIR when set
    for var, sub in (("MPLCONFIGDIR", "mpl"), ("NUMBA_CACHE_DIR", "numba"), ("XDG_CACHE_HOME", "cache"),
                     ("TORCH_HOME", "torch"), ("HF_HOME", "hf")):
        if not os.environ.get(var):
            path = os.path.join(tmp, sub)
            os.makedirs(path, exist_ok=True)
            os.environ[var] = path
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")


set_cache_env()
_raw = os.environ.get("THREADS", "").strip()
_n, _source = resolve_threads(int(_raw) if _raw.isdigit() else 0)
set_thread_env(_n)

from processor.main import main  # noqa: E402

sys.exit(main())
