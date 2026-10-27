"""AdamW explícito: momentos, bias correction y regularización desacoplada."""

# ── SECCIÓN 1. Dependencias y componentes base ─────────────────────────────

import math
import torch

class AdamWManual:
    def __init__(self,params,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01):
        self.params=[p for p in params if p.requires_grad]
        if not self.params:raise ValueError('No hay parámetros entrenables')
        self.lr=lr;self.betas=betas;self.eps=eps;self.weight_decay=weight_decay
        self.m=[torch.zeros_like(p) for p in self.params]
        self.v=[torch.zeros_like(p) for p in self.params]
        self.t=0
    @torch.no_grad()
    def step(self):
        self.t+=1;b1,b2=self.betas
        for i,p in enumerate(self.params):
            if p.grad is None:continue
            g=p.grad
            if not torch.isfinite(g).all():raise FloatingPointError('Gradiente no finito')
            self.m[i].mul_(b1).add_(g,alpha=1-b1)
            self.v[i].mul_(b2).addcmul_(g,g,value=1-b2)
            mhat=self.m[i]/(1-b1**self.t)
            vhat=self.v[i]/(1-b2**self.t)
            # Regularización desacoplada: solamente parámetros con gradiente.
            p.mul_(1-self.lr*self.weight_decay)
            p.addcdiv_(mhat,vhat.sqrt().add_(self.eps),value=-self.lr)
    def zero_grad(self):
        for p in self.params:p.grad=None
    def state_dict(self):
        return {'t':self.t,'m':self.m,'v':self.v,'lr':self.lr,'betas':self.betas,'eps':self.eps,'weight_decay':self.weight_decay}
    def load_state_dict(self,s):
        if len(s['m'])!=len(self.params):raise ValueError('Parámetros incompatibles al reanudar')
        self.t=int(s['t']);self.m=s['m'];self.v=s['v']
        self.lr=s['lr'];self.betas=tuple(s['betas']);self.eps=s['eps'];self.weight_decay=s['weight_decay']
