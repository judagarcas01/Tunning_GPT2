"""Paso 00 — Verificar hardware y dependencias antes de descargar >500 MB."""

# ── SECCIÓN 1. Dependencias y componentes base ─────────────────────────────

from pathlib import Path
import argparse,sys,shutil,platform,importlib.util
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.common import get_config,log,rootpath

# ── SECCIÓN 2. Flujo principal: entradas, cálculos y archivos de salida ────
def main():
    a=argparse.ArgumentParser();a.add_argument('--smoke',action='store_true');a.add_argument('--config');args=a.parse_args()
    c=get_config(args.smoke,args.config)
    import torch
    log(f'Python {platform.python_version()}; Torch {torch.__version__}; backend {c["backend"]}')
    log(f'CPU cores: {__import__("os").cpu_count()} | CUDA: {torch.cuda.is_available()}')
    log(f'Disco libre: {shutil.disk_usage(rootpath(".")).free/1024**3:.2f} GiB')
    try:
        import psutil
        log(f'RAM disponible: {psutil.virtual_memory().available/1024**3:.2f} GiB')
    except ImportError:log('psutil opcional no instalado')
    for name in ['transformers','huggingface_hub','matplotlib','pytest']:
        ok=importlib.util.find_spec(name) is not None
        log(f'{name}: {"OK" if ok else "NO INSTALADO"}')
        if c['backend']=='gpt2' and name in ['transformers','huggingface_hub'] and not ok:
            raise SystemExit(f'Instale requirements.txt antes del modelo real: falta {name}')
    log('Comprobación completada')
if __name__=='__main__':main()
