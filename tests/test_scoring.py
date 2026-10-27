"""Score multi-token sin asumir que POSITIVO/NEGATIVO es un solo token."""
from src.models.toy import ToyGPT,ToyTokenizer
from src.evaluation.label_scoring import predict_one

def test_scoring_with_toy_model():
    tok=ToyTokenizer();m=ToyGPT(tok.vocab_size,160)
    label,scores=predict_one(m,tok,'Muy buen producto de prueba.',160)
    assert label in ('NEGATIVO','POSITIVO')
    assert set(scores)=={'NEGATIVO','POSITIVO'}
