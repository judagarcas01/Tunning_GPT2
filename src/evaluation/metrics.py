"""Métricas binarias escritas explícitamente para no ocultar su formulación."""

# ── SECCIÓN 1. Dependencias y componentes base ─────────────────────────────

from collections import Counter

LABELS=['NEGATIVO','POSITIVO']

def classification_metrics(rows):
    if not rows:raise ValueError('No hay registros para evaluación')
    matrix=[[0,0],[0,0]]
    for row in rows:
        i=LABELS.index(row['label']);j=LABELS.index(row['prediction'])
        matrix[i][j]+=1
    acc=sum(matrix[i][i] for i in range(2))/len(rows)
    per={};f1s=[]
    for i,label in enumerate(LABELS):
        tp=matrix[i][i]
        fp=sum(matrix[k][i] for k in range(2))-tp
        fn=sum(matrix[i][k] for k in range(2))-tp
        precision=tp/(tp+fp) if tp+fp else 0.
        recall=tp/(tp+fn) if tp+fn else 0.
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.
        per[label]={'precision':precision,'recall':recall,'f1':f1,'support':sum(matrix[i])}
        f1s.append(f1)
    return {'accuracy':acc,'macro_f1':sum(f1s)/2,'confusion_matrix':matrix,'per_class':per,'n_evaluated':len(rows)}
