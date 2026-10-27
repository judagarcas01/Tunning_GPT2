"""Entropía cruzada causal manual; nunca pasar labels directamente a transformers."""

# ── SECCIÓN 1. Dependencias y componentes base ─────────────────────────────

import torch
import torch.nn.functional as F

def sft_loss(logits,labels):
    if logits.ndim!=3 or labels.ndim!=2:raise ValueError('Logits [B,T,V], labels [B,T]')
    if logits.shape[:2]!=labels.shape:raise ValueError('Dimensiones incompatibles')
    # logits[:,t] predice tokens[:,t+1]. Respetar desplazamiento causal.
    pred=logits[:,:-1,:].contiguous()
    target=labels[:,1:].contiguous()
    valid=target!=-100
    if valid.sum().item()==0:raise ValueError('No hay tokens supervisados en el batch')
    logp=F.log_softmax(pred,dim=-1)
    safe_targets=target.masked_fill(~valid,0)
    losses=-logp.gather(-1,safe_targets.unsqueeze(-1)).squeeze(-1)
    return (losses*valid).sum()/valid.sum()
