"""Paso 09 — Predicción de reseña nueva: GPT-2 original, full FT y LoRA."""

# ── SECCIÓN 1. Dependencias y componentes base ─────────────────────────────

from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.common import get_config,seed_everything,log
from src.models.load_gpt2 import load_model_tokenizer
from src.models.lora_injection import inject_lora
from src.training.checkpoints import restore_checkpoint
from src.evaluation.label_scoring import predict_one

# ── SECCIÓN 2. Flujo principal: entradas, cálculos y archivos de salida ────
def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--config');p.add_argument('--text',default='La calidad es excelente y estoy muy satisfecho.');a=p.parse_args()
    c=get_config(a.smoke,a.config);seed_everything(c['seed'])
    for kind in ['baseline','full','lora']:
        model,tok=load_model_tokenizer(c)
        if kind=='lora':inject_lora(model,c['lora_rank'],c['lora_alpha'],c['lora_dropout'],c['target_modules'])
        if kind!='baseline':restore_checkpoint(model,c,kind)
        model.eval();pred,scores=predict_one(model,tok,a.text,c['max_length'],c['device'])
        log(f'{kind}: {pred}; logP={scores}')
        del model
if __name__=='__main__':main()
