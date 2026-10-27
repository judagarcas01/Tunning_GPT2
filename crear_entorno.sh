#!/bin/bash

set -e

python3.12 -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip

python -m pip install -r requirements-lock.txt

echo
echo "========================================"
echo "ENTORNO CREADO"
echo "========================================"

python --version

echo
echo "Python:"
which python

echo
echo "TensorFlow GPU:"

python - <<'PY'
import tensorflow as tf

print("TensorFlow:", tf.__version__)
print(
    "GPU:",
    tf.config.list_physical_devices("GPU")
)
PY
