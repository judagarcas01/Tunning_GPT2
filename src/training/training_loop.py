"""Loop de entrenamiento compartido: full y LoRA difieren solo en qué parámetros se entrenan."""

# ── SECCIÓN 1. Dependencias y componentes base ─────────────────────────────

import csv,time
import torch
from src.data.dataloader import loader
from src.training.sft_loss import sft_loss
from src.training.adamw_manual import AdamWManual
from src.utils.common import rootpath,write_json
from src.models.load_gpt2 import print_parameters
from src.utils.hardware import process_ram_mb,cuda_peak_mb
from src.utils.reproducibility import run_environment

def run_training(model,tok,train_records,val_records,cfg,kind):
    device=cfg['device'];model.to(device)
    model.train()
    # Desactivar cache durante backprop y fijar solo parámetros requeridos.
    if hasattr(model,'config'):model.config.use_cache=False
    stats=print_parameters(model)
    optim=AdamWManual(model.parameters(),lr=cfg[f'{kind}_learning_rate'],weight_decay=cfg.get('weight_decay',.01))
    data=loader(train_records,tok,cfg['batch_size'],shuffle=True,seed=cfg['seed'])
    accum=cfg['gradient_accumulation_steps'];max_steps=cfg['max_steps']
    if not train_records or not val_records:raise ValueError('Train/validation deben contener ejemplos')
    history=[];steps=0;epoch=0;start=time.perf_counter();optim.zero_grad()
    peak_rss=process_ram_mb() or 0
    while steps<max_steps:
        epoch+=1
        for batch in data:
            # Mini-batch físico; acumular gradientes promedio antes de actualizar.
            batch={k:v.to(device) for k,v in batch.items()}
            logits=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False).logits
            loss=sft_loss(logits,batch['labels'])
            if not torch.isfinite(loss):raise FloatingPointError('Loss no finita')
            # La última tanda incompleta se divide por su tamaño real, no por accum.
            batch_start=(steps//accum)*accum
            divisor=min(accum,max_steps-batch_start)
            (loss/divisor).backward()
            steps+=1
            # Si max_steps no es múltiplo del factor, se conserva la escala usada.
            if steps%accum==0 or steps==max_steps:
                if cfg.get('grad_clip'):
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],cfg['grad_clip'])
                optim.step();optim.zero_grad()
            rss=process_ram_mb()
            if rss is not None:peak_rss=max(peak_rss,rss)
            history.append({'step':steps,'epoch':epoch,'loss':float(loss.detach().cpu()),'elapsed_sec':round(time.perf_counter()-start,3)})
            if steps==1 or steps%5==0:print(f'{kind}: paso {steps}/{max_steps}, loss={history[-1]["loss"]:.4f}',flush=True)
            if steps>=max_steps:break
    model.eval()
    # Validación de pérdida sobre respuestas; evaluación por etiquetas va en script 07.
    val_losses=[]
    with torch.no_grad():
        for batch in loader(val_records,tok,cfg['batch_size']):
            batch={k:v.to(device) for k,v in batch.items()}
            out=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False).logits
            val_losses.append(float(sft_loss(out,batch['labels']).cpu()))
            if len(val_losses)>=cfg.get('max_validation_batches',4):break
    output=rootpath(cfg['results_dir'])/kind;output.mkdir(parents=True,exist_ok=True)
    with (output/'train_history.csv').open('w',encoding='utf-8',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=['step','epoch','loss','elapsed_sec']);wr.writeheader();wr.writerows(history)
    metrics={'train_last_loss':history[-1]['loss'],'validation_loss':sum(val_losses)/len(val_losses),
             'wall_time_sec':round(time.perf_counter()-start,3),'steps':steps,**stats}
    write_json(metrics,output/'training_summary.json')
    write_json({'peak_process_rss_mb':peak_rss,'peak_cuda_allocated_mb':cuda_peak_mb(device),
                'measurement':'RSS muestreado cada micro-paso; no equivale a memoria total del sistema'},output/'resource_usage.json')
    write_json({'config':{k:v for k,v in cfg.items() if not k.startswith('_')},'environment':run_environment()},
               output/'run_metadata.json')
    return metrics
