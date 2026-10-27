"""Captura configuración de ejecución para comparaciones documentadas."""

# ── SECCIÓN 1. Dependencias y componentes base ─────────────────────────────

import importlib.metadata
import platform
import torch
from src.utils.common import seed_everything

def run_environment():
    packages={}
    for name in ['torch','transformers','huggingface_hub','numpy','psutil','matplotlib']:
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]='no instalado'
    return {'python':platform.python_version(),'platform':platform.platform(),
            'torch_cuda':torch.cuda.is_available(),'packages':packages}
