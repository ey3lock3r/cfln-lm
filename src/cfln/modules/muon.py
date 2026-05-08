import torch


def newton_schulz5_complex(W,n_steps=5):
    m,n=W.shape
    W_r=torch.cat([torch.cat([W.real,-W.imag],dim=1),torch.cat([W.imag,W.real],dim=1)],dim=0)
    norm=W_r.norm(p='fro').clamp(1e-8); X=W_r/norm
    for _ in range(n_steps): A=X@X.T; X=1.5*X-0.5*A@X
    return torch.complex(X[:m,:n],X[m:,:n])


def muon_step(param,buf,name,lr,momentum=0.95,ns_steps=5):
    if param.grad is None: return
    if name not in buf: buf[name]=torch.zeros_like(param.grad)
    buf[name].mul_(momentum).add_(param.grad)
    g_ortho=(newton_schulz5_complex(buf[name],ns_steps) if param.grad.dtype==torch.cfloat
              else _ns5_real(buf[name],ns_steps))
    with torch.no_grad(): param.data.add_(g_ortho,alpha=-lr)
    param.grad=None


def _ns5_real(G,n_steps):
    norm=G.norm(p='fro').clamp(1e-8); X=G/norm
    for _ in range(n_steps): A=X@X.T; X=1.5*X-0.5*A@X
    return X


class MuonOptimizer:
    def __init__(self,params,lr=1e-3,momentum=0.95,ns_steps=5):
        self.params=params; self.lr=lr; self.momentum=momentum; self.ns_steps=ns_steps; self._buf={}
    def step(self,lr=None):
        lr=lr or self.lr
        for name,param in self.params: muon_step(param,self._buf,name,lr,self.momentum,self.ns_steps)
    def zero_grad(self):
        for _,param in self.params:
            if param.grad is not None: param.grad=None
