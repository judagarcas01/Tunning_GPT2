"""Transformer causal mínimo SOLO para ejecutar pruebas locales, NO es GPT-2 preentrenado."""

# ── SECCIÓN 1. Dependencias y componentes base ─────────────────────────────

from types import SimpleNamespace
import string
import torch
from torch import nn
import torch.nn.functional as F

class ToyTokenizer:
    """Codificación por carácter; independiente de Hugging Face, sin BPE real."""
    alphabet = ''.join(sorted(set(string.printable+'áéíóúüñÁÉÍÓÚÜÑ¡¿—€')))
    def __init__(self):
        self.pad_token_id=0;self.eos_token_id=1;self.unk_token_id=2
        self.stoi={c:i+3 for i,c in enumerate(self.alphabet)}
        self.itos={v:k for k,v in self.stoi.items()}
        self.vocab_size=len(self.stoi)+3
        self.pad_token='<PAD>';self.eos_token='<EOS>'
    def encode(self,s,add_special_tokens=False):
        return [self.stoi.get(c,2) for c in s]
    def decode(self,ids,skip_special_tokens=True):
        return ''.join(self.itos.get(int(i),'?') for i in ids if (not skip_special_tokens or int(i)>2))

class ToyConv1D(nn.Module):
    """Replica la convención GPT-2 Conv1D: weight[in_features,out_features]."""
    def __init__(self,in_features,out_features):
        super().__init__()
        self.weight=nn.Parameter(torch.randn(in_features,out_features)*.02)
        self.bias=nn.Parameter(torch.zeros(out_features))
    def forward(self,x):return x@self.weight+self.bias

class ToyAttention(nn.Module):
    def __init__(self,d=32):
        super().__init__()
        self.c_attn=ToyConv1D(d,3*d)
        self.c_proj=ToyConv1D(d,d)
        self.d=d
    def forward(self,x,attention_mask=None):
        b,t,d=x.shape
        q,k,v=self.c_attn(x).chunk(3,dim=-1)
        scores=q@k.transpose(-1,-2)/(d**0.5)
        cm=torch.tril(torch.ones(t,t,device=x.device,dtype=torch.bool))
        scores=scores.masked_fill(~cm,-1e4)
        if attention_mask is not None:
            scores=scores.masked_fill(~attention_mask[:,None,:].bool(),-1e4)
        y=F.softmax(scores,dim=-1)@v
        return self.c_proj(y)

class ToyBlock(nn.Module):
    def __init__(self,d=32):
        super().__init__()
        self.ln_1=nn.LayerNorm(d)
        self.attn=ToyAttention(d)
        self.ln_2=nn.LayerNorm(d)
        self.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(self,x,attention_mask=None):
        x=x+self.attn(self.ln_1(x),attention_mask)
        return x+self.mlp(self.ln_2(x))

class ToyGPT(nn.Module):
    def __init__(self,vocab_size,context=128,d=32):
        super().__init__()
        self.transformer=nn.Module()
        self.transformer.wte=nn.Embedding(vocab_size,d)
        self.transformer.wpe=nn.Embedding(context,d)
        self.transformer.h=nn.ModuleList([ToyBlock(d)])
        self.transformer.ln_f=nn.LayerNorm(d)
        self.lm_head=nn.Linear(d,vocab_size,bias=False)
        self.context=context; self.vocab_size=vocab_size
    def forward(self,input_ids,attention_mask=None,use_cache=False):
        b,t=input_ids.shape
        if t>self.context:raise ValueError('Secuencia excede el contexto de prueba')
        pos=torch.arange(t,device=input_ids.device)
        x=self.transformer.wte(input_ids)+self.transformer.wpe(pos)[None,:,:]
        for block in self.transformer.h:x=block(x,attention_mask)
        logits=self.lm_head(self.transformer.ln_f(x))
        return SimpleNamespace(logits=logits)
