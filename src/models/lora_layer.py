"""LoRA manual sobre Conv1D de GPT-2: W0[in,out], delta=(X A)B."""

# ── SECCIÓN 1. Dependencias y componentes base ─────────────────────────────

import math
import torch
from torch import nn
import torch.nn.functional as F

class LoRAConv1D(nn.Module):
    def __init__(self,base,r=4,alpha=8,dropout=0.0):
        super().__init__()
        if r<1:raise ValueError('El rango r debe ser positivo')
        self.base=base
        for p in self.base.parameters():p.requires_grad_(False)
        d_in,d_out=base.weight.shape
        self.A=nn.Parameter(torch.empty(d_in,r,device=base.weight.device,dtype=base.weight.dtype))
        self.B=nn.Parameter(torch.zeros(r,d_out,device=base.weight.device,dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.A.T,a=math.sqrt(5))
        self.scale=alpha/r
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        return self.base(x)+((self.dropout(x)@self.A)@self.B)*self.scale
