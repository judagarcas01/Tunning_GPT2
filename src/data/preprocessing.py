"""Enmascarar prompt (-100), alinear BPE y evitar truncación de etiqueta."""

# ── SECCIÓN 1. Dependencias y componentes base ─────────────────────────────

from src.utils.common import read_jsonl

PROMPT_PREFIX='Clasifica la valoración de la siguiente reseña como POSITIVO o NEGATIVO.\nReseña: '
PROMPT_SUFFIX='\nSentimiento:'

def format_prompt(review):return PROMPT_PREFIX+review+PROMPT_SUFFIX

def tokenized_record(record,tokenizer,max_length):
    answer=' '+record['label']
    text=record['text']
    # Nunca truncar la respuesta; acortar solo texto de la reseña si hace falta.
    # Importante: BPE puede fusionar tokens en los límites. Se valida prefijo real.
    for _ in range(10):
        prompt=format_prompt(text)
        pre=tokenizer.encode(prompt,add_special_tokens=False)
        full=tokenizer.encode(prompt+answer,add_special_tokens=False)
        if full[:len(pre)]!=pre:
            raise ValueError('BPE fusionó caracteres entre prompt y respuesta; cambie la plantilla o añada separador seguro.')
        eos=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        if len(full)+len(eos)<=max_length:break
        new=max(0,int(len(text)*.72))
        if new==len(text) or not text:raise ValueError('max_length insuficiente para la plantilla')
        text=text[:new]
    else:raise ValueError('max_length insuficiente después de truncar reseña')
    ids=full+eos
    labels=[-100]*len(pre)+full[len(pre):]+eos
    if not any(v!=-100 for v in labels):raise ValueError('Respuesta vacía')
    return {'id':record['id'],'text':text,'label':record['label'], 'prompt':prompt,'input_ids':ids,'labels':labels}

def build_tokenized(split_path,tokenizer,max_length):
    return [tokenized_record(item,tokenizer,max_length) for item in read_jsonl(split_path)]
