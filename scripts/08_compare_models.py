"""Paso 08 — Consolidar métricas reales, historias de loss y gráficos."""

# ── SECCIÓN 1. Dependencias y componentes base ─────────────────────────────

from pathlib import Path
import argparse,sys,csv
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.common import get_config,rootpath,read_json,log
from src.evaluation.plots import make_plots

# ── SECCIÓN 2. Flujo principal: entradas, cálculos y archivos de salida ────
def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--config');a=p.parse_args()
    c=get_config(a.smoke,a.config);output=rootpath(c['results_dir']);rows=[]
    for kind in ['baseline','full','lora']:
        path=output/kind/'metrics.json'
        if not path.exists():raise FileNotFoundError(f'Evalúe antes de comparar: {path}')
        m=read_json(path)
        train=output/kind/'training_summary.json'
        extra=read_json(train) if train.exists() else {}
        rows.append({'method':kind,'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],
                     'n_evaluated':m['n_evaluated'],'smoke':m['smoke'],
                     'trainable_parameters':extra.get('parameters_trainable',0),
                     'training_seconds':extra.get('wall_time_sec',0)})
    folder=output/'comparison';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'metrics_comparison.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    make_plots(c)
    for row in rows:log(str(row))
    log(f'Archivo comparativo: {folder/"metrics_comparison.csv"}')
if __name__=='__main__':main()
