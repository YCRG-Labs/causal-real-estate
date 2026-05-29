"""Side-effect-only module: silence the noisy sklearn `UserWarning: X does not
have valid feature names` that fires every time we call `predict` on a
numpy array after LightGBM was fit on a DataFrame.

Imported (often as `import _silence  # noqa: F401`) by every entry-point
script in this directory to keep stdout/stderr clean for paste.
"""
import warnings

# the sklearn validation warning that floods nuisance-fit logs
warnings.filterwarnings(
    "ignore",
    message=".*does not have valid feature names.*",
    category=UserWarning,
)
# a few other low-value warnings that show up in our pipeline
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
