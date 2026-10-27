"""Limpieza conservadora: no eliminar negaciones, acentos ni puntuación."""

# ── SECCIÓN 1. Dependencias y componentes base ─────────────────────────────

import re,unicodedata

def clean_text(s):
    if not isinstance(s,str): return ''
    s=unicodedata.normalize('NFC',s)
    s=re.sub(r'\s+',' ',s).strip()
    return s

def dedup_key(s):
    return clean_text(s).casefold()

def map_label(v):
    try:v=int(v)
    except (ValueError,TypeError):return None
    return 'NEGATIVO' if v==0 else 'POSITIVO' if v==4 else None
