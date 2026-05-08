import torch
import torch.nn as nn
from cfln.utils import to_real, tanh_c


class ComplexSTIHead(nn.Module):
    """STI prediction head. v5.9.3: slice fix + memory cap."""
    def __init__(self,d_c,S=32,D_g=8,vocab_size=None,beta_U=0.3,D_bptt=8):
        super().__init__()
        self.D_g=D_g; self.D_bptt=D_bptt; self.beta_U=beta_U; self.S=S
        self.C_proj  =nn.Parameter(torch.randn(d_c,dtype=torch.cfloat)/d_c**0.5)
        self.w_c_g   =nn.Parameter(torch.zeros(D_g+1,dtype=torch.cfloat))
        self.B_c_out =nn.Parameter((torch.randn(d_c,S)+1j*torch.randn(d_c,S)).to(torch.cfloat)/S**0.5)
        self.W_vocab =nn.Linear(2*d_c,vocab_size) if vocab_size else None
        self._ocn_hist=[]; self._ocn_buf_det=[]

    def step_and_predict(self,x_c,U=None):
        j_out=(self.C_proj.conj()@x_c.mean(0)).sum()
        n_h=len(self._ocn_hist); n_d=len(self._ocn_buf_det)
        if n_h+n_d>=self.D_g:
            if n_h>=self.D_g: hist=torch.stack(self._ocn_hist[-self.D_g:])
            else:
                old=torch.stack(self._ocn_buf_det[-(self.D_g-n_h):]); rec=torch.stack(self._ocn_hist)
                hist=torch.cat([old,rec],dim=0)
            z_next=tanh_c(self.w_c_g[0]+(self.w_c_g[1:self.D_g+1]*hist.flip(0)).sum()+j_out)
        else: z_next=tanh_c(self.w_c_g[0]+j_out)
        self._ocn_hist.append(z_next)
        if len(self._ocn_hist)>self.D_bptt:
            self._ocn_buf_det.append(self._ocn_hist.pop(0).detach())
            if len(self._ocn_buf_det)>self.D_bptt: self._ocn_buf_det=self._ocn_buf_det[-self.D_bptt:]
        needed=self.S-len(self._ocn_hist)
        det_use=(torch.stack(self._ocn_buf_det[-needed:]) if needed>0 and self._ocn_buf_det
                  else torch.zeros(0,dtype=torch.cfloat,device=x_c.device))
        rec_use=torch.stack(self._ocn_hist); total=det_use.shape[0]+rec_use.shape[0]
        if total<self.S:
            Z_pred=torch.cat([torch.zeros(self.S-total,dtype=torch.cfloat,device=x_c.device),det_use,rec_use])
        else: Z_pred=torch.cat([det_use,rec_use])
        B=x_c.shape[0]; X_hat=(self.B_c_out@Z_pred.unsqueeze(-1)).squeeze(-1).unsqueeze(0).expand(B,-1)
        logits=self.W_vocab(to_real(X_hat)) if self.W_vocab else None
        unc_w=((1.0-self.beta_U*U).clamp(0.1) if U is not None else torch.ones(B,device=x_c.device))
        return logits,unc_w

    def reset(self): self._ocn_hist.clear(); self._ocn_buf_det.clear()
