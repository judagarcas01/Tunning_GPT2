"""Padding manual: etiquetas -100 e attention_mask=0 para posiciones PAD."""

# ── SECCIÓN 1. Dependencias y componentes base ─────────────────────────────

import torch
from torch.utils.data import DataLoader

def collate(rows,pad_id):
    L=max(len(r['input_ids']) for r in rows)
    ids=[];labels=[];masks=[]
    for r in rows:
        n=L-len(r['input_ids'])
        ids.append(r['input_ids']+[pad_id]*n)
        labels.append(r['labels']+[-100]*n)
        masks.append([1]*len(r['input_ids'])+[0]*n)
    return {'input_ids':torch.tensor(ids,dtype=torch.long),
            'labels':torch.tensor(labels,dtype=torch.long),
            'attention_mask':torch.tensor(masks,dtype=torch.long)}

def loader(records,tok,batch_size,shuffle=False,seed=42):
    # Un generador dedicado evita que la inicialización de LoRA altere el orden
    # de las reseñas respecto del full fine-tuning.
    generator=torch.Generator().manual_seed(seed)
    return DataLoader(records,batch_size=batch_size,shuffle=shuffle,generator=generator,
                      collate_fn=lambda rows:collate(rows,tok.pad_token_id))
