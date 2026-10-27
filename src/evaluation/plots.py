"""Figuras publicables sin asumir que un smoke-test represente resultados científicos."""

# ── SECCIÓN 1. Dependencias y componentes base ─────────────────────────────

import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils.common import read_json,rootpath

def make_plots(cfg):
    results=rootpath(cfg['results_dir']); dest=results/'comparison';dest.mkdir(parents=True,exist_ok=True)
    data=[]
    for kind in ['baseline','full','lora']:
        path=results/kind/'metrics.json'
        if path.exists():data.append((kind,read_json(path)))
    if data:
        fig,ax=plt.subplots(figsize=(6,3.5))
        labels=[k for k,_ in data]
        acc=[m['accuracy'] for _,m in data]; f1=[m['macro_f1'] for _,m in data]
        xs=list(range(len(labels))); width=.36
        ax.bar([x-width/2 for x in xs],acc,width,label='Accuracy')
        ax.bar([x+width/2 for x in xs],f1,width,label='Macro-F1')
        ax.set_xticks(xs,labels);ax.set_ylim(0,1);ax.legend();ax.set_ylabel('Métrica')
        fig.tight_layout();fig.savefig(dest/'performance.png',dpi=160);plt.close(fig)
    # Una figura por matriz: filas=etiqueta real, columnas=predicción.
    for kind,metrics in data:
        matrix=metrics['confusion_matrix']
        fig,ax=plt.subplots(figsize=(4,3.5))
        im=ax.imshow(matrix)
        for i,row in enumerate(matrix):
            for j,count in enumerate(row):
                ax.text(j,i,str(count),ha='center',va='center',color='white' if count>max(max(r) for r in matrix)/2 else 'black')
        ax.set_xticks([0,1],['NEG','POS']);ax.set_yticks([0,1],['NEG','POS'])
        ax.set_xlabel('Predicción');ax.set_ylabel('Etiqueta real');ax.set_title(kind)
        fig.tight_layout();fig.savefig(dest/f'confusion_{kind}.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3.5));any_hist=False
    for kind in ['full','lora']:
        path=results/kind/'train_history.csv'
        if not path.exists():continue
        with path.open(newline='',encoding='utf-8') as h: records=list(csv.DictReader(h))
        if records:
            ax.plot([int(r['step']) for r in records],[float(r['loss']) for r in records],label=kind)
            any_hist=True
    if any_hist:
        ax.set_xlabel('Paso');ax.set_ylabel('SFT loss');ax.legend();fig.tight_layout()
        fig.savefig(dest/'learning_curves.png',dpi=160)
    plt.close(fig)
