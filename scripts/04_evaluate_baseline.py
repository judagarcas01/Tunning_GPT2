"""Paso 04 — Baseline sin entrenar: log P(POSITIVO/NEGATIVO | reseña)."""

# ── SECCIÓN 1. Dependencias y componentes base ─────────────────────────────

from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.common import get_config,seed_everything
from src.models.load_gpt2 import load_model_tokenizer
from src.utils.execution import evaluate

# ── SECCIÓN 2. Flujo principal: entradas, cálculos y archivos de salida ────
def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--config');p.add_argument('--limit',type=int);a=p.parse_args()
    c=get_config(a.smoke,a.config);seed_everything(c['seed'])
    model,tok=load_model_tokenizer(c)
    evaluate(model,tok,c,'baseline',a.limit)
if __name__=='__main__':main()
