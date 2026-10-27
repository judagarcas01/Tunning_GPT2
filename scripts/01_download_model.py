"""Paso 01 — Descargar pesos GPT-2 y tokenizer; o crear modelo toy reproducible."""

# ── SECCIÓN 1. Dependencias y componentes base ─────────────────────────────

from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.common import get_config,rootpath,log,seed_everything,write_json

# ── SECCIÓN 2. Flujo principal: entradas, cálculos y archivos de salida ────
def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--config');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    cfg=get_config(a.smoke,a.config);dest=rootpath(cfg['model_dir'])
    if a.dry_run:
        log(f'Se descargaría {cfg["model_repo"]}@{cfg["model_revision"]} en {dest}');return
    dest.mkdir(parents=True,exist_ok=True)
    if cfg['backend']=='toy':
        import torch
        from src.models.toy import ToyTokenizer,ToyGPT
        seed_everything(cfg['seed'])
        tok=ToyTokenizer();model=ToyGPT(tok.vocab_size,cfg['max_length'])
        torch.save(model.state_dict(),dest/'toy_weights.pt')
        log(f'Modelo de juguete aleatorio guardado en {dest}. NO es GPT-2.')
    else:
        from huggingface_hub import snapshot_download
        # Excluimos 1GB de pesos TensorFlow y Flax; solo PyTorch + tokenizer BPE.
        snapshot_download(repo_id=cfg['model_repo'],revision=cfg['model_revision'],local_dir=str(dest),
             allow_patterns=['config.json','pytorch_model.bin','model.safetensors','*.safetensors.index.json',
                            'tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json','generation_config.json'])
        if not (dest/'config.json').exists():raise RuntimeError('Falta config.json')
        if not any((dest/k).exists() for k in ['pytorch_model.bin','model.safetensors']):raise RuntimeError('Faltan pesos PyTorch')
        log(f'Modelo real descargado en {dest}')
    write_json({'repo':cfg['model_repo'],'revision':cfg['model_revision'],'backend':cfg['backend']},dest/'manifest.json')
if __name__=='__main__':main()
