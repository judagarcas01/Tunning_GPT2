"""Sustituir sólo c_attn; no se importan librerías PEFT."""

# ── SECCIÓN 1. Dependencias y componentes base ─────────────────────────────

from src.models.lora_layer import LoRAConv1D

def inject_lora(model,r=4,alpha=8,dropout=0.0,target_modules=('c_attn',)):
    for p in model.parameters():p.requires_grad_(False)
    count=0
    for block in model.transformer.h:
        for name in target_modules:
            # Primero únicamente c_attn; c_proj también admite LoRA si se desea.
            layer=getattr(block.attn,name,None)
            if layer is None or not hasattr(layer,'weight'):raise ValueError(f'Capa de atención no encontrada: {name}')
            if isinstance(layer,LoRAConv1D):raise ValueError('LoRA ya insertado')
            setattr(block.attn,name,LoRAConv1D(layer,r,alpha,dropout))
            count+=1
    if count==0:raise ValueError('No se insertaron capas LoRA')
    return count

def adapter_state(model):
    return {name:p.detach().cpu().clone() for name,p in model.named_parameters() if name.endswith('.A') or name.endswith('.B')}

def load_adapter_state(model,state):
    keys=set(adapter_state(model))
    if keys!=set(state):raise ValueError(f'Llaves de adaptador diferentes: {keys.symmetric_difference(set(state))}')
    named=dict(model.named_parameters())
    for name,tensor in state.items():
        if named[name].shape!=tensor.shape:raise ValueError(f'Shape incorrecto: {name}')
        import torch
        with torch.no_grad():
            named[name].copy_(tensor.to(named[name].device))
