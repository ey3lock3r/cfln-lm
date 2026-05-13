"""CFLNConfig — all hyperparameters for CFLN v6.0.9.

Defaults follow docs/CFLN_Master_Spec.md (v9.0).
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class CFLNConfig:
    # ── Core dimensions ───────────────────────────────────────────────────
    d_c: int = 256                  # complex feature dimension (v6.0.9: 256)
    vocab_size: int = 32_000        # vocabulary size
    L: int = 6                      # CFL-5 stack depth

    # ── CNEP memory bank ──────────────────────────────────────────────────
    n_l: int = 2_112               # local units (2048 + 64 for global-tier removal)
    n_p: int = 128                  # persistent units (v6.0.9: 128)
    d_e_l: int = 32                 # local energy projection dim
    d_e_p: int = 64                 # persistent energy projection dim
    # d_e_g removed in v6.0.8 — global tier removed

    # ── LRU / SSM ─────────────────────────────────────────────────────────
    d_ssm_fast: int = 32            # LRU SSM dim
    S_f: int = 32                   # LRU readout dim
    per_sequence_memory: bool = True

    # ── Chunking ──────────────────────────────────────────────────────────
    C_chunk: int = 512              # tokens per chunk (v6.0.9: 512)

    # ── Titans ────────────────────────────────────────────────────────────
    eta_init: float = 0.01          # Titans learning rate (eta_titans in spec)
    theta_decay_init: float = 0.99
    null_threshold_init: float = 0.95
    k_null: float = 50.0            # null-update sharpness
    beta_null_aux: float = 0.01

    # ── Domain detection ──────────────────────────────────────────────────
    domain_alpha: float = 0.90
    domain_mag_alpha: float = 0.99
    domain_threshold_init: float = 3.0
    surprise_warmup_chunks: int = 32
    slow_drift_window: int = 500
    slow_drift_threshold: float = 0.5

    # ── Telescoping buffers ───────────────────────────────────────────────
    K_L1: int = 128                 # telescoping L1 buffer
    K_L2: int = 32                  # telescoping L2 buffer
    K_L3: int = 32                  # telescoping L3 buffer
    beta_telescoping: float = 1.0

    # ── Surprise archive ──────────────────────────────────────────────────
    N_archive: int = 256            # surprise archive capacity
    surprise_N_tau: int = 100
    surprise_threshold_pct: float = 0.80

    # ── LISTA ─────────────────────────────────────────────────────────────
    N_iter: int = 4                 # LISTA iterations (task spec; spec table uses N_iter_refine=8)
    N_iter_refine: int = 8          # full LISTA refinement iterations
    N_hop_refine: int = 4           # Hopfield coupling interval
    use_hopfield_refine: bool = True
    use_escape_refine: bool = True
    lambda_lista: float = 0.1       # L_lista weight
    n_layers_diff: int = 2          # pre-LISTA CFL layers
    lista_min_ratio: float = 0.25
    lista_convergence_ratio: float = 0.5
    delta_stuck: float = 0.1        # LISTA escape trigger
    delta_min: float = 0.01         # LISTA escape min norm
    epsilon_esc: float = 0.05       # LISTA escape noise scale

    # ── Sparse code / episodic rule caches ───────────────────────────────
    K_sparse: int = 32              # sparse code cache size (sparse_code_cache_K)
    N_rules: int = 256              # ARC episodic rule cache size (v9.0: raised from 64)

    # ── Coactivation register ─────────────────────────────────────────────
    K_hebb: int = 16                # coactivation register size

    # ── Top-k routing ─────────────────────────────────────────────────────
    k_l: int = 40                   # top-k local units selected per step (legacy; adaptive in v9.0)
    k_l_min: int = 10               # §1.64 C2: adaptive k_l lower bound
    k_l_max: int = 40               # §1.64 C2: adaptive k_l upper bound

    # ── CRoPE ─────────────────────────────────────────────────────────────
    rope_base: float = 5_250_000.0  # NTK-scaled for 1M-token context
    L_train: int = 2_048            # rope_L_train
    L_target: int = 1_048_576       # rope_L_target
    use_crope: bool = True

    # ── Node Fourier reservoir ─────────────────────────────────────────────
    d_r_node: int = 8               # per-unit reservoir dimension
    rho_node: float = 0.95          # spectral radius
    rho_fast: float = 0.85          # fast mode spectral radius
    rho_mid: float = 0.90           # medium mode spectral radius
    rho_slow: float = 0.99          # slow mode spectral radius

    # ── LISTA session reservoir ────────────────────────────────────────────
    d_r_lista: int = 128            # LISTA reservoir dim = d_c//2 per §1.13
    rho_lista: float = 0.99

    # ── GAT ───────────────────────────────────────────────────────────────
    n_heads_gat: int = 4

    # ── Diffusion / noise conditioning ───────────────────────────────────
    n_fourier: int = 32             # Fourier noise conditioning frequencies
    T_diff: int = 200               # diffusion steps
    D_g: int = 8                    # H_c_l history depth

    # ── CTP (Complex Thinking Protocol) ───────────────────────────────────
    think_threshold: float = 0.6    # uncertainty threshold to trigger thinking
    tau_think: float = 0.5          # thinking token loss weight
    n_think_max: int = 16           # max thinking tokens (task spec)
    max_think_tokens: int = 64      # alias used by generate function

    # ── Optimiser ─────────────────────────────────────────────────────────
    lr_local: float = 1e-3          # AdamW local params (lr_unit in spec)
    lr_persist: float = 1e-6        # Stiefel W_p learning rate
    lr_muon: float = 1e-3
    lr_muon_diff: float = 1e-4
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    lr_start: float = 1e-3          # cosine schedule start
    lr_end: float = 1e-4            # cosine schedule end
    lr_global: float = 3e-4         # AdamW scalar params
    lr_unit: float = 1e-3           # unit params incl. log_scale_l
    grad_clip: float = 1.0
    schedule_grad_clip: float = 0.5  # lam_p_schedule clip
    gradient_checkpointing: bool = False
    grad_accum_steps: int = 1

    # ── SI (Synaptic Intelligence) ────────────────────────────────────────
    c_SI: float = 0.1               # SI regularization weight (task spec)
    rho_SI: float = 0.999           # SI omega decay
    beta_SI: float = 3.0            # SI per-unit lr sharpness
    si_warmup_steps: int = 100
    min_snapshot_interval: int = 50
    si_proactive_threshold: float = 0.8
    proactive_cooldown: int = 20

    # ── Alpha-freeze ──────────────────────────────────────────────────────
    sensory_fraction: float = 0.15
    alpha_freeze_percentile: float = 0.85

    # ── Dormancy ──────────────────────────────────────────────────────────
    N_dormant: int = 512

    # ── BPTT / misc ───────────────────────────────────────────────────────
    D_bptt: int = 4                 # BPTT depth for OCN
    K_stats: int = 4                # stats buffer size
    lambda_compress: float = 0.001  # VQ reconstruction loss (v9.0 §1.51: 0.001)

    # ── SE-1 k-shot refinement (§1.70 C7) ────────────────────────────────
    tau_proto_min: float = 0.4      # U_epi_cal gate threshold for SE-1 accumulation
    K_proto_max: int = 10           # max proto accumulations per young unit

    # ── Fisher-KL continual learning (§1.57) ─────────────────────────────
    beta_KL: float = 0.5            # Fisher-KL penalty weight
    beta_KL_warmup: int = 500       # steps before KL penalty activates
    beta_SI_stiefel: float = 0.25   # §E.2: SI sharpness for Stiefel params (spec: 0.25, not 3.0)

    # ── v9.0 loss weights ─────────────────────────────────────────────────
    lambda_vq: float = 0.01         # VQ-Telescope commitment loss
    lambda_bridge: float = 0.1      # L_bridge predictive coding loss
    lambda_diversity: float = 0.01  # beam anti-collapse diversity loss
    lambda_lipschitz: float = 0.001 # ROB-L young unit Lipschitz reg
    lambda_sigma_reg: float = 0.001 # ROB-S learned sigma binding reg
    lambda_prec: float = 0.001      # precision weighting regularization
    lambda_mlm: float = 0.3         # SE-2 MDLM masked LM loss

    # ── MDLM (§1.44 SE-2) ────────────────────────────────────────────────
    p_mask: float = 0.15            # masking probability for MDLM

    # ── Role binding (§1.55) ──────────────────────────────────────────────
    n_roles: int = 8                # number of role vectors

    # ── Micro-consolidation (§1.54 KA-MC) ────────────────────────────────
    alpha_micro: float = 0.0001     # per-chunk ARC→CNEP consolidation rate
    alpha_young: float = 0.1        # activity threshold for "young" unit

    # ── Memory threshold dict ─────────────────────────────────────────────
    memory_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "eps_s": 0.01,
        "eps_p": 0.001,
        "eps_split": 0.5,
        "eps_merge": 0.95,
        "r_reset": 0.3,
        "eps_H": 1e-4,
    })

    # ── v9.0 beam search ──────────────────────────────────────────────────────
    beam_B_max: int = 3             # §1.66: max beam width (B_eff = 1 + prev_U_meta*(B_max-1))

    # ── Training stage ────────────────────────────────────────────────────
    stage: int = 0                  # training phase (0 = Phase 0 skeleton)

    # ── Batch / sequence dims ─────────────────────────────────────────────
    T: int = 2048                   # sequence length for training
    B: int = 8                      # batch size
