"""Side-effect-only module: silence the noisy sklearn `UserWarning: X does not
have valid feature names` that fires every time we call `predict` on a
numpy array after LightGBM was fit on a DataFrame.

Also defuses the macOS multi-OpenMP-runtime crash. PyTorch (libomp bundled
with libtorch_cpu), LightGBM (Homebrew libomp), and sklearn (_openmp_helpers
libomp) each load a separate libomp.dylib into the same process; when import
order interleaves them on macOS 26.5+, LightGBM's DatasetLoader hits a null
kmp_info during pthread_create and segfaults at __kmp_suspend_initialize_thread.
KMP_DUPLICATE_LIB_OK lets the runtimes coexist; OMP_NUM_THREADS=1 stops them
from racing for the same physical cores. Joblib's loky workers carry their
own thread caps so this does not slow the outer bootstrap.

Imported (often as `import _silence  # noqa: F401`) by every entry-point
script in this directory to keep stdout/stderr clean for paste.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*does not have valid feature names.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*The least populated class in y.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*Trying to unpickle estimator.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*'force_all_finite' was renamed.*",
    category=FutureWarning,
)
