"""Paso 06 — Reinicia el modelo BASE y aprende exclusivamente matrices LoRA manuales."""

# ── SECCIÓN 1. Dependencias y componentes base ─────────────────────────────

from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.common import get_config,seed_everything,rootpath,read_jsonl,log
from src.models.load_gpt2 import load_model_tokenizer
from src.models.lora_injection import inject_lora
from src.training.training_loop import run_training
from src.training.checkpoints import save_checkpoint

# ── SECCIÓN 2. Flujo principal: entradas, cálculos y archivos de salida ────
def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--config');p.add_argument('--steps',type=int);a=p.parse_args()
    c=get_config(a.smoke,a.config);seed_everything(c['seed']);c['max_steps']=a.steps or c['max_steps']
    model,tok=load_model_tokenizer(c)
    n=inject_lora(model,c['lora_rank'],c['lora_alpha'],c['lora_dropout'],c['target_modules'])
    log(f'Adaptadores agregados: {n}; el peso base permanece congelado')
    train=list(read_jsonl(rootpath(c['tokenized_dir'])/'train.jsonl'))
    val=list(read_jsonl(rootpath(c['tokenized_dir'])/'validation.jsonl'))
    summary=run_training(model,tok,train,val,c,'lora')
    saved=save_checkpoint(model,c,'lora',summary);log(f'Solo matrices LoRA: {saved}')
if __name__=='__main__':main()
