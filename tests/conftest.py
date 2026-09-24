"""pytest setup shared by every test file (loaded before any test module, so before torch is imported).

- No test may use the GPU: the laptop GPU runs other jobs. On Windows an EMPTY environment variable is dropped, so
  CUDA_VISIBLE_DEVICES="" does not hide the GPU from the CUDA driver; an empty or unset value becomes -1 here.
- The `slow` marker (tests that load the real 2B teacher) is registered; deselect with -m "not slow".
"""
import os

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: loads the real 2B teacher from the HF cache (deselect with -m 'not slow')")
