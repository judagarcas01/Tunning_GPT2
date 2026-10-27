"""Muestreo estratificado; conserva particiones train/validation/test originales."""

# ── SECCIÓN 1. Dependencias y componentes base ─────────────────────────────

import random
from src.data.cleaning import clean_text,dedup_key,map_label
from src.utils.common import read_jsonl

def select_splits(raw_dir,n_per_class,seed=42):
    seen=set(); out={}; audit={}
    # Orden fijo y deduplicación inter-particiones sin usar clases del test para entrenar.
    for ix,split in enumerate(['train','validation','test']):
        groups={'NEGATIVO':[],'POSITIVO':[]}; scanned=0; duplicates=0
        path=raw_dir/f'{split}.jsonl'
        for item in read_jsonl(path):
            scanned+=1
            label=map_label(item.get('label')); text=clean_text(item.get('text'))
            if label is None or len(text)<8:continue
            key=dedup_key(text)
            if key in seen:
                duplicates+=1; continue
            seen.add(key)
            groups[label].append({'id':str(item.get('id',f'{split}_{scanned}')),'text':text,'label':label,'label_original':int(item['label'])})
        rng=random.Random(seed+ix)
        target=n_per_class[split]; selected=[]
        for label, rows in groups.items():
            if len(rows)<target:
                raise ValueError(f'{split}/{label}: {len(rows)} reseñas; solicitadas {target}. Reduzca samples_per_class.')
            selected+=rng.sample(rows,target)
        rng.shuffle(selected);out[split]=selected
        audit[split]={'raw_scanned':scanned,'duplicates_removed':duplicates,'per_class':target,'selected':len(selected)}
    return out,audit
