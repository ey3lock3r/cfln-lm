import heapq
import torch


class SurpriseArchive:
    """Importance-based archive. v6.0.6: cosine dedup prevents near-duplicate slots."""
    def __init__(self,d_c,N_archive=256,N_tau=100,W_warmup=32,tau_percentile=0.80,
                 tau_sa_dedup=0.85):
        self.d_c=d_c; self.N_archive=N_archive; self.N_tau=N_tau
        self.W_warmup=W_warmup; self.tau_pct=tau_percentile
        self.tau_sa_dedup=tau_sa_dedup       # v6.0.6: cosine dedup threshold
        self.entries=torch.zeros(d_c,N_archive,dtype=torch.cfloat)
        self.surprises=torch.zeros(N_archive); self._heap=[]; self._n_filled=0
        self._surprise_history=torch.zeros(N_tau); self._hist_ptr=0; self._hist_fill=0; self._chunk_count=0

    def to(self,device): self.entries=self.entries.to(device); self.surprises=self.surprises.to(device); return self

    @torch.no_grad()
    def update_threshold(self,s_t):
        self._surprise_history[self._hist_ptr]=s_t
        self._hist_ptr=(self._hist_ptr+1)%self.N_tau; self._hist_fill=min(self._hist_fill+1,self.N_tau)

    def get_threshold(self):
        if self._hist_fill<2: return 0.0
        valid=self._surprise_history[:self._hist_fill]; idx=int(self.tau_pct*self._hist_fill)
        return float(torch.sort(valid).values[min(idx,self._hist_fill-1)].item())

    @torch.no_grad()
    def maybe_add(self,c_k,s_t):
        self._chunk_count+=1; self.update_threshold(s_t)
        if self._chunk_count<=self.W_warmup: return False
        tau=self.get_threshold()
        if s_t<=tau: return False
        # v6.0.6 dedup: check cosine similarity to existing entries
        if self._n_filled>0:
            Xi=self.entries[:,:self._n_filled]  # (d_c, n_filled)
            c_norm=c_k/(c_k.norm().clamp(1e-8))
            sims=((c_norm@Xi.conj()).real/(Xi.norm(dim=0).clamp(1e-8)))  # (n_filled,)
            best_sim,best_slot=sims.max(0)
            if float(best_sim)>self.tau_sa_dedup:
                # Update existing similar entry instead of inserting duplicate
                slot_idx=int(best_slot.item())
                self.entries[:,slot_idx]=(0.7*self.entries[:,slot_idx]+0.3*c_k)
                self.surprises[slot_idx]=max(float(self.surprises[slot_idx]),s_t)
                return True
        if self._n_filled<self.N_archive:
            slot=self._n_filled; self.entries[:,slot]=c_k; self.surprises[slot]=s_t
            heapq.heappush(self._heap,(s_t,slot)); self._n_filled+=1; return True
        s_min,slot_min=self._heap[0]
        if s_t>s_min:
            self.entries[:,slot_min]=c_k; self.surprises[slot_min]=s_t
            heapq.heapreplace(self._heap,(s_t,slot_min)); return True
        return False

    def retrieve(self,x_c_query,beta=1.0):
        if self._n_filled==0: return torch.zeros_like(x_c_query)
        Xi=self.entries[:,:self._n_filled].to(x_c_query.device)
        w=torch.softmax((x_c_query@Xi.conj()).real*beta,dim=-1).to(torch.cfloat)
        return w@Xi.mH

    def reset(self):
        with torch.no_grad(): self.entries.zero_(); self.surprises.zero_()
        self._heap=[]; self._n_filled=0; self._hist_ptr=0; self._hist_fill=0
        self._chunk_count=0; self._surprise_history.zero_()
