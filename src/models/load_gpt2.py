"""Cargar GPT-2 real SIN Trainer, o Transformer de juguete para smoke-test."""

# ── SECCIÓN 1. Dependencias y componentes base ─────────────────────────────

from pathlib import Path
import torch
from .toy import ToyTokenizer,ToyGPT
from src.utils.common import rootpath

def load_model_tokenizer(cfg,device=None):
    device=device or cfg.get('device','cpu')
    if cfg.get('backend')=='toy':
        path=rootpath(cfg['model_dir'])/'toy_weights.pt'
        if not path.exists():raise FileNotFoundError(f'Primero ejecute scripts/01_download_model.py --smoke: {path}')
        tok=ToyTokenizer()
        model=ToyGPT(tok.vocab_size,context=cfg['max_length'])
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    else:
        # ÚNICA dependencia de transformers: reconstrucción de GPT-2 y BPE.
        from transformers import AutoModelForCausalLM,AutoTokenizer
        base=rootpath(cfg['model_dir'])
        if not (base/'config.json').exists():raise FileNotFoundError(f'Descargue primero GPT-2: {base}')
        tok=AutoTokenizer.from_pretrained(str(base),local_files_only=True,use_fast=True)
        # Algunos GPT-2 no definen padding. Usar EOS como PAD (solo la máscara decide).
        if tok.pad_token_id is None:tok.pad_token=tok.eos_token
        model=AutoModelForCausalLM.from_pretrained(str(base),local_files_only=True)
        model.config.pad_token_id=tok.pad_token_id
    model=model.to(device)
    return model,tok

def print_parameters(model):
    total=sum(p.numel() for p in model.parameters())
    train=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parámetros total={total:,}; entrenables={train:,}; proporción={train/max(total,1):.4%}')
    return {'parameters_total':total,'parameters_trainable':train}
