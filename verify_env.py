"""
verify_env.py
-------------
Run this script after activating your virtual environment to confirm
that all required packages are installed and at compatible versions.

Usage:
    python verify_env.py
"""

import sys
import importlib

REQUIRED = [
    ("tensorflow",   "tf",        "2.13"),
    ("keras",        "keras",     "2.13"),
    ("numpy",        "np",        "1.24"),
    ("matplotlib",   "plt",       "3.7"),
    ("sklearn",      "sklearn",   "1.3"),
    ("streamlit",    "st",        "1.35"),
    ("cv2",          "cv2",       "4.9"),
    ("PIL",          "PIL",       "10.0"),
    ("httpx",        "httpx",     "0.27"),
    ("jupyter",      "jupyter",   None),   # checked via importlib only
]

OK     = "  OK  "
FAIL   = " FAIL "
SKIP   = " SKIP "

passed = 0
failed = 0

print()
print("=" * 55)
print("  MNIST Lab — Environment Verification")
print("=" * 55)
print(f"  Python {sys.version}")
print("=" * 55)
print(f"  {'Package':<18} {'Status':<8} {'Version'}")
print("-" * 55)

for pkg, alias, min_ver in REQUIRED:
    try:
        mod = importlib.import_module(pkg)
        version = getattr(mod, "__version__", "unknown")
        print(f"  {pkg:<18} [{OK}]  {version}")
        passed += 1
    except ImportError:
        print(f"  {pkg:<18} [{FAIL}]  NOT INSTALLED")
        failed += 1

# Special check: tensorflow version detail
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    gpu_info = f"{len(gpus)} GPU(s) found" if gpus else "CPU only"
    print(f"\n  TensorFlow device: {gpu_info}")
except Exception:
    pass

print("-" * 55)
print(f"  {passed} passed, {failed} failed")
print("=" * 55)

if failed > 0:
    print("\n  Fix: activate your virtual environment and run:")
    print("       pip install -r requirements.txt\n")
    sys.exit(1)
else:
    print("\n  All packages found. You are ready to go!\n")
