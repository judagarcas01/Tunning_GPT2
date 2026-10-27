"""LoRA: salida inicial idéntica, base congelada, gradientes y checkpoints."""
import torch
from src.models.toy import ToyGPT,ToyTokenizer
from src.models.lora_injection import inject_lora,adapter_state,load_adapter_state

def test_lora_equivalence_gradients_and_state():
    torch.manual_seed(42)
    t=ToyTokenizer();m=ToyGPT(t.vocab_size,160)
    m.eval();ids=torch.randint(3,t.vocab_size,(2,12))
    before=m(ids).logits.detach().clone()
    assert inject_lora(m,2,4)==1
    after=m(ids).logits
    assert torch.allclose(before,after,atol=1e-6)
    assert all(not p.requires_grad for n,p in m.named_parameters() if not (n.endswith('.A') or n.endswith('.B')))
    after.square().mean().backward()
    assert any(n.endswith('.B') and p.grad is not None and p.grad.abs().sum()>0 for n,p in m.named_parameters())
    state=adapter_state(m)
    load_adapter_state(m,state)
    assert set(state)==set(adapter_state(m))
