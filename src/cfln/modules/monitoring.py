import torch
from collections import deque


class SlowDriftDetector:
    def __init__(self,window=500,threshold=0.5,N_check=200):
        self.window=window; self.threshold=threshold; self.N_check=N_check
        self._history=deque(maxlen=window); self._baseline_mean=None; self._last_check=-9999
    def update(self,s_domain_ema,step):
        self._history.append(s_domain_ema)
        if step-self._last_check<self.N_check: return False
        self._last_check=step
        if len(self._history)<self.window//4: return False
        current_mean=sum(self._history)/len(self._history)
        if self._baseline_mean is None: self._baseline_mean=current_mean; return False
        if self._baseline_mean>1e-8:
            drift=abs(current_mean-self._baseline_mean)/self._baseline_mean
            if drift>self.threshold: self._baseline_mean=current_mean; return True
        return False
    def reset(self): self._history.clear(); self._baseline_mean=None


class DocumentStreamingContext:
    """v5.9.4: begin_document resets BOTH node reservoir AND LISTA reservoir."""
    def __init__(self,model,window_size=256,stride=None):
        self.model=model; self.window_size=window_size; self.stride=stride or window_size
        self._active=False; self._chunk_count=0; self._window_count=0

    def begin_document(self):
        self.model.telescoping_mem.reset()
        self.model.surprise_archive.reset()
        self.model.encoder.titans.reset_to_neutral()
        self.model.encoder.fast_lru.reset()
        self.model.sti_head.reset()
        self.model._pos_offset=0
        self.model.bank.reset_reservoir()                   # NEW v5.9.4: node reservoir
        self.model.diff_aux.cun.reset_lista_reservoir()     # NEW v5.9.4: LISTA reservoir
        # Reset EMA baselines so U_epistemic/U_temporal from previous doc don't bleed in
        self.model.bank._e_min_ema.fill_(1.0)
        self.model.bank._h_route_ema.fill_(1.0)
        self.model.bank._ema_delta_bank.fill_(1e-6)
        self._active=True; self._chunk_count=0; self._window_count=0

    def end_document(self): self._active=False
    @property
    def is_active(self): return self._active
    def record_window(self,n_chunks): self._chunk_count+=n_chunks; self._window_count+=1

    @staticmethod
    def build_windows(doc_ids,window_size=256,stride=None):
        stride=stride or window_size; N=doc_ids.shape[-1]; windows=[]
        for start in range(0,N-1,stride):
            end=min(start+window_size,N)
            if end-start>=4: windows.append(doc_ids[...,start:end])
        return windows


class NeedleInHaystackEvaluator:
    """Tiered evaluation matching memory levels. Unchanged from v5.9.3."""
    def __init__(self,model,vocab_size=4096,C_chunk=32):
        self.model=model; self.V=vocab_size; self.C=C_chunk; self.needle_start=vocab_size//2

    @torch.no_grad()
    def evaluate(self,distances,n_trials=50,device=None):
        device=device or next(self.model.parameters()).device; results={}
        for D in distances:
            hits=sum(self._single_trial(D,device) for _ in range(n_trials))
            results[D]=hits/n_trials; print(f"  Distance {D:5d}: acc={results[D]:.3f}")
        return results

    def _single_trial(self,distance,device):
        a=torch.randint(self.needle_start,self.V-2,(1,))
        b_val=self.needle_start+((a.item()-self.needle_start+1)%(self.V//2-4))
        b=torch.tensor([b_val])
        filler1=torch.randint(2,self.needle_start,(distance,))
        filler2=torch.randint(2,self.needle_start,(distance,))
        seq=torch.cat([filler1,a,b,filler2,a.clone()]).unsqueeze(0).to(device)
        self.model.eval(); self.model.reset_for_inference()
        ctx=DocumentStreamingContext(self.model,window_size=min(256,seq.shape[1]))
        ctx.begin_document()
        if seq.shape[1]<=256:
            logits,_,_=self.model(seq,training=False); pred=logits[0,-1,:].argmax().item()
        else:
            windows=DocumentStreamingContext.build_windows(seq,window_size=128)
            last_logits=None
            for w in windows: logits,_,_=self.model(w,training=False); last_logits=logits
            pred=last_logits[0,-1,:].argmax().item() if last_logits is not None else -1
        ctx.end_document(); return int(pred==b.item())

    def run_full_eval(self,device,C_chunk=None):
        C=C_chunk or self.C; distances=[C//2,C*4,C*16,C*64,C*128]
        print(f"\nNeedle-in-Haystack Tiered (C_chunk={C}):")
        return self.evaluate(distances,n_trials=50,device=device)
