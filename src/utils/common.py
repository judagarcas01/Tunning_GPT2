"""Rutas, configuración, semillas y E/S compartidas; sin datos ocultos ni servicios externos."""

# ── SECCIÓN 1. Dependencias y componentes base ─────────────────────────────

import json
import random
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]

def config_path(smoke=False, path=None):
    return Path(path).resolve() if path else ROOT / 'configs' / ('smoke.json' if smoke else 'base.json')

def get_config(smoke=False, path=None):
    with config_path(smoke,path).open(encoding='utf-8') as fh:
        c=json.load(fh)
    c['_smoke']=smoke or (c.get('backend')=='toy')
    return c

def rootpath(value):
    return ROOT/value

def ensure_parent(p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);return p

def write_json(obj,p):
    p=ensure_parent(p)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def read_jsonl(p):
    with Path(p).open(encoding='utf-8') as fh:
        for i,line in enumerate(fh,1):
            if line.strip():
                try: yield json.loads(line)
                except json.JSONDecodeError as ex: raise ValueError(f'{p}:{i}: JSON inválido') from ex

def write_jsonl(rows,p):
    with ensure_parent(p).open('w',encoding='utf-8') as fh:
        for row in rows: fh.write(json.dumps(row,ensure_ascii=False)+'\n')

def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def log(msg): print(f'[FT-Doctorado] {msg}',flush=True)

# Al cargar scripts desde cualquier directorio, usar PYTHONPATH o bootstrap del propio script.
def boot_script():
    if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
