"""Paso 02 — Dataset oficial JSONL o datos sintéticos EXCLUSIVOS para smoke-test."""

# ── SECCIÓN 1. Dependencias y componentes base ─────────────────────────────

from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.common import get_config,rootpath,log,write_jsonl,write_json

# ── SECCIÓN 2. Flujo principal: entradas, cálculos y archivos de salida ────
def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--config');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    cfg=get_config(a.smoke,a.config);dest=rootpath(cfg['raw_dir'])
    if a.dry_run:
        log(f'Se descargarían train.jsonl, validation.jsonl, test.jsonl desde {cfg["dataset_repo"]} hacia {dest}');return
    dest.mkdir(parents=True,exist_ok=True)
    if cfg['backend']=='toy':
        # Nunca tratar estos ejemplos como métricas reales.
        for split,n in [('train',12),('validation',6),('test',6)]:
            records=[]
            for k in range(n):
                records.append({'id':f'synthetic_{split}_bad_{k}','text':f'Este producto sintético malo numero {k}: falla y no funciona.', 'label':0,'label_text':'0'})
                records.append({'id':f'synthetic_{split}_good_{k}','text':f'Este producto sintético bueno numero {k}: excelente y funciona.', 'label':4,'label_text':'4'})
            # textos entre particiones necesitan ser diferentes para deduplicación global
            for row in records:row['text']=f'{split}: '+row['text']
            write_jsonl(records,dest/f'{split}.jsonl')
            log(f'Datos sintéticos {split}: {len(records)} registros')
    else:
        from huggingface_hub import hf_hub_download
        for split in ['train','validation','test']:
            filepath=hf_hub_download(repo_id=cfg['dataset_repo'],repo_type='dataset',revision=cfg['dataset_revision'],
                            filename=f'{split}.jsonl',local_dir=str(dest))
            log(f'{split}: {filepath}')
    write_json({'repo':cfg['dataset_repo'],'revision':cfg['dataset_revision'],'backend':cfg['backend']},dest/'manifest.json')
if __name__=='__main__':main()
