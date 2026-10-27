"""Verificar limpieza, muestreo y máscara del prompt."""
import pytest
from src.data.cleaning import clean_text,map_label
from src.data.preprocessing import tokenized_record
from src.models.toy import ToyTokenizer

def test_cleaning():
    assert clean_text(' No   funciona \n')=='No funciona'
    assert map_label(0)=='NEGATIVO' and map_label(4)=='POSITIVO'
    assert map_label(2) is None

def test_masks_and_lengths():
    tok=ToyTokenizer()
    r=tokenized_record({'id':'x','label':'NEGATIVO','text':'Muy malo el producto.'},tok,160)
    assert len(r['input_ids'])==len(r['labels'])
    assert -100 in r['labels'] and any(x!=-100 for x in r['labels'])
    assert max(r['input_ids'])<tok.vocab_size
