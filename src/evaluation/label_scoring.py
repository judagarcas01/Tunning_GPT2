"""Inferencia restringida a dos etiquetas: log P(etiqueta | prompt)."""

# ── SECCIÓN 1. Dependencias y componentes base ─────────────────────────────

import torch
import torch.nn.functional as F
from src.data.preprocessing import format_prompt

@torch.no_grad()
def label_log_probability(model,tokenizer,review,label,max_length,device='cpu'):
    suffix=' '+label
    text=review
    for _ in range(12):
        prompt=format_prompt(text)
        p=tokenizer.encode(prompt,add_special_tokens=False)
        full=tokenizer.encode(prompt+suffix,add_special_tokens=False)
        if full[:len(p)]!=p:raise ValueError('BPE no preserva frontera de prompt en inferencia')
        if len(full)<=max_length:break
        if not text:raise ValueError('Prompt supera max_length')
        text=text[:int(len(text)*.75)]
    else:raise ValueError('Imposible ajustar longitud del prompt')
    ids=torch.tensor([full],dtype=torch.long,device=device)
    logits=model(input_ids=ids,use_cache=False).logits[0]
    logp=F.log_softmax(logits,dim=-1)
    # Logits de posición k-1 predicen token k; no normalizamos por longitud.
    return sum(logp[k-1,full[k]].item() for k in range(len(p),len(full)))

@torch.no_grad()
def predict_one(model,tok,text,max_length,device='cpu'):
    labels=['NEGATIVO','POSITIVO']
    # MISMO prefijo para las dos etiquetas: sin este control, la truncación
    # dependiente de la longitud de la etiqueta sesgaría la comparación.
    from src.data.preprocessing import format_prompt
    common=text
    for _ in range(12):
        prompt_ids=tok.encode(format_prompt(common),add_special_tokens=False)
        candidate_ids=[tok.encode(format_prompt(common)+' '+s,add_special_tokens=False) for s in labels]
        if all(ids[:len(prompt_ids)]==prompt_ids for ids in candidate_ids) and all(len(ids)<=max_length for ids in candidate_ids):break
        if not common:raise ValueError('Las etiquetas no caben en max_length')
        common=common[:int(len(common)*.75)]
    else:raise ValueError('No fue posible usar un prefijo común')
    scores={s:label_log_probability(model,tok,common,s,max_length,device) for s in labels}
    return max(scores,key=scores.get),scores
