"""Comparación AdamW manual vs torch.optim.AdamW en un paso."""
import torch
from src.training.adamw_manual import AdamWManual

def test_adamw_matches_reference():
    a=torch.nn.Parameter(torch.tensor([1.,-2.]))
    b=torch.nn.Parameter(a.detach().clone())
    o=AdamWManual([a],lr=1e-3,weight_decay=.01)
    ref=torch.optim.AdamW([b],lr=1e-3,weight_decay=.01)
    a.grad=torch.tensor([.2,-.5]); b.grad=a.grad.clone()
    o.step();ref.step()
    assert torch.allclose(a,b,atol=1e-7)
