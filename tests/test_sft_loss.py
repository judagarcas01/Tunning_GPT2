"""Verificación de la pérdida sobre tokens supervisados y del gradiente."""
import torch
from src.training.sft_loss import sft_loss

def test_masked_positions_do_not_change_loss():
    torch.manual_seed(1)
    logits=torch.randn(1,5,7,requires_grad=True)
    labels=torch.tensor([[-100,-100,-100,3,2]])
    value=sft_loss(logits,labels)
    other=logits.detach().clone();other[:,0,:]+=50  # posición 0 predice label[1]=-100
    assert torch.allclose(value,sft_loss(other,labels))
    value.backward()
    assert logits.grad[:,0,:].abs().sum().item()==0
    assert logits.grad[:,2:4,:].abs().sum().item()>0

def test_no_valid_targets_raises():
    try:sft_loss(torch.randn(1,3,4),torch.full((1,3),-100));assert False
    except ValueError:pass
