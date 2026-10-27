"""Cargar splits tokenizados / evaluar con logits de GPT-2 sin predicciones inventadas."""

# ── SECCIÓN 1. Dependencias y componentes base ─────────────────────────────

import csv,time
from src.utils.common import read_jsonl,rootpath,write_json,log
from src.evaluation.label_scoring import predict_one
from src.evaluation.metrics import classification_metrics

def processed(cfg,split):
    p=rootpath(cfg['processed_dir'])/f'{split}.jsonl'
    if not p.exists():raise FileNotFoundError(f'Falta {p}. Ejecute 03_prepare_data.py')
    return list(read_jsonl(p))

def evaluate(model,tok,cfg,kind,limit=None):
    import torch
    model.eval();records=processed(cfg,'test')
    if limit is None:limit=cfg.get('eval_limit',30)
    if limit>0:records=records[:limit]
    if not records:raise ValueError('Test vacío')
    t0=time.perf_counter();preds=[]
    for idx,row in enumerate(records,1):
        pred,scores=predict_one(model,tok,row['text'],cfg['max_length'],cfg['device'])
        preds.append({'id':row['id'],'text':row['text'],'label':row['label'],'prediction':pred,
                      'score_negative':scores['NEGATIVO'],'score_positive':scores['POSITIVO']})
        if idx%10==0:log(f'{kind}: evaluadas {idx}/{len(records)} reseñas')
    scores=classification_metrics(preds);scores.update({'method':kind,'elapsed_sec':round(time.perf_counter()-t0,3),
                  'evaluation_subset':len(records),'smoke':cfg.get('backend')=='toy'})
    folder=rootpath(cfg['results_dir'])/kind;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'predictions.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(preds[0]));w.writeheader();w.writerows(preds)
    write_json(scores,folder/'metrics.json')
    log(f'{kind}: accuracy={scores["accuracy"]:.3f}, macro-F1={scores["macro_f1"]:.3f} (n={len(preds)})')
    return scores
