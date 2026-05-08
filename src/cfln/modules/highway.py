import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexMHCHighway(nn.Module):
    """n_hc=2 mHC. v5.9.3: _get_params cached within forward pass."""
    def __init__(self, d_c, L=6):
        super().__init__()
        self.d_c=d_c; self.L=L; in_dim=4*d_c
        self.w_b=nn.ParameterList([nn.Parameter(torch.zeros(in_dim)) for _ in range(L)])
        self.w_a=nn.ParameterList([nn.Parameter(torch.zeros(in_dim,2)) for _ in range(L)])
        self.w_c=nn.ParameterList([nn.Parameter(torch.zeros(in_dim,2)) for _ in range(L)])
        self.s_b=nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for _ in range(L)])
        self.s_a=nn.ParameterList([nn.Parameter(torch.tensor([1.0,0.0])) for _ in range(L)])
        self.s_c=nn.ParameterList([nn.Parameter(torch.tensor([1.0,0.0])) for _ in range(L)])
        self.alpha_b=nn.ParameterList([nn.Parameter(torch.tensor(0.01)) for _ in range(L)])
        self.alpha_a=nn.ParameterList([nn.Parameter(torch.tensor(0.01)) for _ in range(L)])
        self.alpha_c=nn.ParameterList([nn.Parameter(torch.tensor(0.01)) for _ in range(L)])
        self._param_cache=None

    def _get_params(self,l,xf,xs):
        key=(id(xf),id(xs),l)
        if self._param_cache is not None and self._param_cache[0]==key:
            return self._param_cache[1],self._param_cache[2],self._param_cache[3]
        flat=F.rms_norm(torch.cat([xf.real.mean(0),xf.imag.mean(0),
                                    xs.real.mean(0),xs.imag.mean(0)]),[4*self.d_c])
        b=torch.sigmoid(self.alpha_b[l]*(flat@self.w_b[l])+self.s_b[l])
        A=torch.softmax(self.alpha_a[l]*(flat@self.w_a[l])+self.s_a[l],dim=0)
        C=torch.sigmoid(self.alpha_c[l]*(flat@self.w_c[l])+self.s_c[l])/2.0
        self._param_cache=(key,b,A,C); return b,A,C

    def inject(self,xf,xs,l): _,A,_=self._get_params(l,xf,xs); return A[0]*xf+A[1]*xs
    def update(self,xf,xs,f_out,l):
        b,_,C=self._get_params(l,xf,xs)
        return (1-b)*xf+b*xs+C[0]*f_out, b*xf+(1-b)*xs+C[1]*f_out
    def init_streams(self,B,device):
        z=torch.zeros(B,self.d_c,dtype=torch.cfloat,device=device); return z.clone(),z.clone()
