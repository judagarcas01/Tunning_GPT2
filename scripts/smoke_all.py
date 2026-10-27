"""Ejecutar en orden TODAS las etapas con toy-model, registrar código de salida y tiempo."""

# ── SECCIÓN 1. Dependencias y componentes base ─────────────────────────────

from pathlib import Path
import os,subprocess,sys,time,json
ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=['00_check_environment','01_download_model','02_download_reviews','03_prepare_data','04_evaluate_baseline',
         '05_train_full','06_train_lora','07_evaluate_models','08_compare_models','09_predict']

# ── SECCIÓN 2. Flujo principal: entradas, cálculos y archivos de salida ────
def main():
    env=os.environ.copy()
    env.setdefault('OMP_NUM_THREADS','2')
    env.setdefault('MKL_NUM_THREADS','2')
    env.setdefault('OPENBLAS_NUM_THREADS','1')
    results=[]
    for name in SCRIPTS:
        cmd=[sys.executable,str(ROOT/'scripts'/f'{name}.py'),'--smoke']
        t=time.perf_counter()
        r=subprocess.run(cmd,cwd=ROOT,env=env,text=True,capture_output=True)
        elapsed=round(time.perf_counter()-t,3)
        print(f'{name}: {"OK" if r.returncode==0 else "FALLÓ"}, {elapsed}s',flush=True)
        if r.returncode:print(r.stdout,r.stderr,file=sys.stderr)
        results.append({'script':name,'exit_code':r.returncode,'seconds':elapsed,'stdout_tail':r.stdout[-1000:],'stderr_tail':r.stderr[-1000:]})
        if r.returncode:break
    dest=ROOT/'results'/'smoke'/'smoke_report.json';dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Informe:',dest)
    if any(item['exit_code'] for item in results):sys.exit(1)
if __name__=='__main__':main()
