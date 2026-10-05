"""pytest setup shared by every test file (loaded before any test module, so before torch is imported).

- No test may use the GPU: the laptop GPU runs other jobs. On Windows an EMPTY environment variable is dropped, so
  CUDA_VISIBLE_DEVICES="" does not hide the GPU from the CUDA driver; an empty or unset value becomes -1 here.
- The `slow` marker (tests that load the real 2B teacher) is registered; deselect with -m "not slow".
- In parallel (pytest-xdist, requirements-dev.txt: `python -m pytest tests -n 8 --dist loadfile`), each worker runs
  its torch and BLAS pools on KITSUNE_TEST_THREADS threads (default 1): N workers of one thread each instead of N pools
  the size of the machine. --dist loadfile keeps a file's tests on one worker, so its module fixtures (the synthetic
  corpora, stores and tiny runs) are built once, as in a serial run.
"""
import os

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if os.environ.get("PYTEST_XDIST_WORKER"):
    _threads = os.environ.get("KITSUNE_TEST_THREADS", "1")
    for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_k] = _threads


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: loads the real 2B teacher from the HF cache (deselect with -m 'not slow')")
