"""Paso 07 — Comparación sobre el MISMO test. Modelos separados desde checkpoint base."""

# ── SECCIÓN 1. Dependencias y componentes base ─────────────────────────────

from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.common import get_config,seed_everything,log
from src.models.load_gpt2 import load_model_tokenizer
from src.models.lora_injection import inject_lora
from src.training.checkpoints import restore_checkpoint
from src.utils.execution import evaluate

# ── SECCIÓN 2. Flujo principal: entradas, cálculos y archivos de salida ────
def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--config');p.add_argument('--limit',type=int);a=p.parse_args()
    c=get_config(a.smoke,a.config);seed_everything(c['seed'])
    for kind in ['baseline','full','lora']:
        model,tok=load_model_tokenizer(c)
        if kind=='lora':inject_lora(model,c['lora_rank'],c['lora_alpha'],c['lora_dropout'],c['target_modules'])
        if kind!='baseline':restore_checkpoint(model,c,kind)
        evaluate(model,tok,c,kind,a.limit)
        del model
    log('Evaluación de tres métodos completada')
if __name__=='__main__':main()
