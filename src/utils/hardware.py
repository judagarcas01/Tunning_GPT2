"""Métricas de recursos observadas; no confundir memoria RSS con RAM total del sistema."""

# ── SECCIÓN 1. Dependencias y componentes base ─────────────────────────────

import os
import torch

def process_ram_mb():
    try:
        import psutil
        return round(psutil.Process(os.getpid()).memory_info().rss/1024**2,2)
    except ImportError:
        return None

def cuda_peak_mb(device):
    if str(device).startswith('cuda') and torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated()/1024**2,2)
    return None
