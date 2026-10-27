"""Nunca sobrescribir modelo base. Guardar pesos (full) o solo adaptadores (LoRA)."""

# ── SECCIÓN 1. Dependencias y componentes base ─────────────────────────────

from pathlib import Path
import torch
from src.models.lora_injection import adapter_state,load_adapter_state
from src.utils.common import ensure_parent,write_json,read_json

def checkpoint_paths(cfg,kind):
    from src.utils.common import rootpath
    folder=rootpath(cfg['checkpoints_dir'])/kind
    return folder,folder/'weights.pt',folder/'metadata.json'

def save_checkpoint(model,cfg,kind,metrics=None):
    folder,weights,meta=checkpoint_paths(cfg,kind)
    folder.mkdir(parents=True,exist_ok=True)
    state=adapter_state(model) if kind=='lora' else {k:v.detach().cpu() for k,v in model.state_dict().items()}
    torch.save(state,weights)
    write_json({'kind':kind,'model_repo':cfg['model_repo'],'model_revision':cfg.get('model_revision','main'),
                'backend':cfg['backend'],'rank':cfg['lora_rank'] if kind=='lora' else None,
                'alpha':cfg['lora_alpha'] if kind=='lora' else None,'targets':cfg['target_modules'] if kind=='lora' else None,
                'seed':cfg['seed'],'metrics':metrics or {}},meta)
    return weights

def restore_checkpoint(model,cfg,kind):
    _,weights,meta=checkpoint_paths(cfg,kind)
    if not weights.exists():raise FileNotFoundError(f'Sin checkpoint {kind}: {weights}')
    metadata=read_json(meta)
    if metadata['model_repo']!=cfg['model_repo'] or metadata['backend']!=cfg['backend']:
        raise ValueError('Checkpoint generado desde una base diferente')
    state=torch.load(weights,map_location='cpu',weights_only=True)
    if kind=='lora':load_adapter_state(model,state)
    else:model.load_state_dict(state)
    return metadata
