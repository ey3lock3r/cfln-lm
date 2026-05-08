import math
import torch
import torch.nn as nn


class ComplexUncertaintyModule(nn.Module):
    def __init__(self,d_c):
        super().__init__()
        self.W_unc=nn.Parameter((torch.randn(d_c,d_c)+1j*torch.randn(d_c,d_c)).to(torch.cfloat)/d_c**0.5)
        self.log_beta_unc=nn.Parameter(torch.tensor(0.0))
    def forward(self,x_c_final,Z_L):
        x_c_head=x_c_final@self.W_unc.conj().T
        U=(1.0-torch.exp(-torch.exp(self.log_beta_unc)*Z_L.float())).clamp(0,1)
        return x_c_head,U


class PerLayerLamPSchedule(nn.Module):
    def __init__(self,L,lam_p_min=0.01,lam_p_max=0.5):
        super().__init__()
        self.L=L
        self.log_lam_p=nn.ParameterList([nn.Parameter(torch.tensor(math.log(
            lam_p_min+(lam_p_max-lam_p_min)*l/max(L-1,1)))) for l in range(L)])
    def get_lam_p(self,l): return torch.exp(self.log_lam_p[l])
    def forward(self,_): return torch.stack([torch.exp(p) for p in self.log_lam_p])


class CFLNPathologyMonitor:
    def __init__(self,L,monitor_freq=100):
        self.L=L; self.monitor_freq=monitor_freq; self.E_D_history=[]
    def step(self,step,layer_outputs,**kwargs):
        if step%self.monitor_freq!=0: return {}
        E_D=[float(info.get('s_l',None).std().item()) if info.get('s_l') is not None else 0.0
              for info in layer_outputs]
        self.E_D_history.append((step,E_D))
        if len(self.E_D_history)>1000: self.E_D_history=self.E_D_history[-1000:]
        return {'E_D_per_layer':E_D}


class UncertaintyCurriculumSampler:
    def __init__(self,dataset_size,decay=0.9,temperature=1.0):
        self.N=dataset_size; self.decay=decay; self.temperature=temperature
        self.uncertainty_ema=torch.ones(dataset_size)
    @torch.no_grad()
    def update(self,seq_ids,uncertainties):
        for b in range(len(seq_ids)):
            idx=int(seq_ids[b].item())
            if 0<=idx<self.N:
                self.uncertainty_ema[idx]=(self.decay*self.uncertainty_ema[idx]
                                            +(1-self.decay)*float(uncertainties[b].item()))
    def get_indices(self,batch_size):
        n_uni=batch_size//2; n_pri=batch_size-n_uni; uni=torch.randint(0,self.N,(n_uni,))
        lg=self.uncertainty_ema/self.temperature; wts=torch.softmax(lg-lg.max(),dim=0)
        return torch.cat([uni,torch.multinomial(wts,n_pri,replacement=True)],dim=0)
