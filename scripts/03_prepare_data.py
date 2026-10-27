"""Paso 03 — Deduplicación, balance, prompts y tensores para SFT manual."""

# ── SECCIÓN 1. Dependencias y componentes base ─────────────────────────────

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.common import get_config,rootpath,write_jsonl,write_json,log,seed_everything
from src.data.sampling import select_splits
from src.data.preprocessing import tokenized_record
from src.models.load_gpt2 import load_model_tokenizer

# ── SECCIÓN 2. Flujo principal: entradas, cálculos y archivos de salida ────
def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--config');a=p.parse_args()
    cfg=get_config(a.smoke,a.config);seed_everything(cfg['seed'])
    splits,audit=select_splits(rootpath(cfg['raw_dir']),cfg['samples_per_class'],cfg['seed'])
    processed=rootpath(cfg['processed_dir']);tokenized=rootpath(cfg['tokenized_dir'])
    processed.mkdir(parents=True,exist_ok=True);tokenized.mkdir(parents=True,exist_ok=True)
    # El tokenizador se carga del modelo local; modelo se libera sin entrenarlo.
    model,tok=load_model_tokenizer(cfg);del model
    for split,rows in splits.items():
        write_jsonl(rows,processed/f'{split}.jsonl')
        sequences=[tokenized_record(row,tok,cfg['max_length']) for row in rows]
        write_jsonl(sequences,tokenized/f'{split}.jsonl')
        log(f'{split}: {len(rows)} reseñas, secuencias <= {cfg["max_length"]} tokens')
    write_json({'dataset':cfg['dataset_repo'],'seed':cfg['seed'],'audit':audit},processed/'data_manifest.json')
if __name__=='__main__':main()
