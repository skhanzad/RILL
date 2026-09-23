"""DecisionFlow model.

Architecture (F5-style conditional flow matching, adapted to typed decisions):

  text branch (run once per request)
    token embeddings -> bottom L_txt pretrained bidirectional-encoder layers (block-isolated attention)
                     -> N ConvNeXt V2 blocks (segment-masked depthwise conv + per-segment GRN; zero-init residual)
  answer track (one scalar per option slot, flow state x_t, and a masked self-condition c)
    [h_text ; phi(x_t, c)] -> top L_dit pretrained layers wrapped as DiT blocks
                              (adaLN scale/shift/gate from the flow time t, identity at init)
                           -> final modulation -> linear -> option logits z
  readout ("decoder")
    p = softmax(z) per question;  x1_hat = p (yes/no, choice) or cumsum(p) (score, CDF encoding)
    predicted flow v = (x1_hat - x_t) / (1 - t)

Training minimises || x1_hat - x_1 ||^2 (the (1-t)^2-reweighted flow-matching loss, i.e. a denoising Brier score) plus
cross-entropy. At t = 0 the Bayes-optimal x1_hat is E[x_1 | request] = the answer distribution itself.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb

from .layout import Batch, OPT_TOKEN, Q_TOKEN

# cuDNN SDPA rebuilds its execution graph for every new (batch, length) shape, which variable-length decision batches
# trigger constantly; the memory-efficient kernel handles arbitrary boolean masks at the same steady-state speed.
if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)


# ----------------------------------------------------------------------------------------------------------------------
# segment helpers (questions have different numbers of options; slots are stored flat, question by question)

def seg_softmax(z: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    zmax = torch.full((n,), -torch.inf, device=z.device, dtype=z.dtype).scatter_reduce(0, seg, z, "amax", include_self=True)
    e = torch.exp(z - zmax[seg])
    s = torch.zeros(n, device=z.device, dtype=z.dtype).index_add(0, seg, e)
    return e / s[seg]


def seg_log_softmax(z: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    zmax = torch.full((n,), -torch.inf, device=z.device, dtype=z.dtype).scatter_reduce(0, seg, z, "amax", include_self=True)
    zc = z - zmax[seg]
    lse = torch.log(torch.zeros(n, device=z.device, dtype=z.dtype).index_add(0, seg, torch.exp(zc)))
    return zc - lse[seg]


def seg_cumsum(x: torch.Tensor, q_start: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
    """Cumulative sum inside each question (slots of one question are contiguous)."""
    c = torch.cumsum(x, 0)
    offset = torch.where(q_start > 0, c[(q_start - 1).clamp_min(0)], torch.zeros_like(c[q_start]))
    return c - offset[seg]


def seg_sum(x: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    return torch.zeros(n, device=x.device, dtype=x.dtype).index_add(0, seg, x)


# ----------------------------------------------------------------------------------------------------------------------

def build_masks(seg: torch.Tensor, pos: torch.Tensor, valid: torch.Tensor, window: int):
    """Boolean (B,1,L,L) masks (True = may attend) for global and sliding-window layers.
    State tokens (segment 0) read only the state; question tokens read the state and their own question."""
    B, L = seg.shape
    sq, sk = seg[:, :, None], seg[:, None, :]
    allowed = valid[:, None, :] & valid[:, :, None] & ((sq == sk) | ((sk == 0) & (sq > 0)))
    ar = torch.arange(L, device=seg.device)
    eye = (ar[:, None] == ar[None, :])[None]          # (ONNX-friendly form of torch.eye(L, dtype=bool))
    glob = allowed | eye
    near = (pos[:, :, None] - pos[:, None, :]).abs() <= window
    loc = (allowed & near) | eye
    return glob[:, None], loc[:, None]


def attention(attn: nn.Module, x: torch.Tensor, mask: torch.Tensor, cos_sin, dropout: float = 0.0) -> torch.Tensor:
    B, L, _ = x.shape
    H = attn.config.num_attention_heads
    D = attn.head_dim
    qkv = attn.Wqkv(x).view(B, L, 3, H, D)
    q, k, v = qkv.unbind(dim=2)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    cos, sin = cos_sin
    q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout, scale=D ** -0.5)
    o = o.transpose(1, 2).reshape(B, L, H * D)
    return attn.out_drop(attn.Wo(o))


class ConvNeXtV2Block1d(nn.Module):
    """ConvNeXt V2 block over the token axis: depthwise conv (masked so it never crosses a segment boundary) ->
    LayerNorm -> pointwise expand -> GELU -> GRN (statistics per segment) -> pointwise project; zero-init residual."""

    def __init__(self, d: int, kernel: int = 7, mult: int = 2):
        super().__init__()
        self.r = kernel // 2
        self.dw = nn.Parameter(torch.zeros(kernel, d))
        nn.init.normal_(self.dw, std=0.02)
        self.dw_b = nn.Parameter(torch.zeros(d))
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Linear(d, mult * d)
        self.gamma = nn.Parameter(torch.zeros(mult * d))
        self.beta = nn.Parameter(torch.zeros(mult * d))
        self.pw2 = nn.Linear(mult * d, d)
        nn.init.zeros_(self.pw2.weight); nn.init.zeros_(self.pw2.bias)

    def forward(self, h: torch.Tensor, seg: torch.Tensor, valid: torch.Tensor, gid: torch.Tensor, ngroups: int):
        B, L, d = h.shape
        y = h * self.dw[self.r]
        for s in range(1, self.r + 1):
            # neighbours at distance s on both sides, only when they belong to the same segment
            same_r = torch.zeros_like(valid); same_r[:, :-s] = (seg[:, s:] == seg[:, :-s]) & valid[:, s:]
            same_l = torch.zeros_like(valid); same_l[:, s:] = (seg[:, :-s] == seg[:, s:]) & valid[:, :-s]
            right = F.pad(h[:, s:], (0, 0, 0, s)) * same_r[..., None]
            left = F.pad(h[:, :-s], (0, 0, s, 0)) * same_l[..., None]
            y = y + right * self.dw[self.r + s] + left * self.dw[self.r - s]
        y = y + self.dw_b
        y = F.gelu(self.pw1(self.norm(y)))
        # Global Response Normalisation, with the "global" L2 norm taken over each segment separately
        yf = y.float() * valid[..., None]
        sq = torch.zeros(ngroups, y.shape[-1], device=y.device, dtype=torch.float32).index_add_(0, gid.reshape(-1), yf.reshape(-1, y.shape[-1]) ** 2)
        g = (sq + 1e-6).sqrt()
        n = g / (g.mean(-1, keepdim=True) + 1e-6)
        n = n[gid.reshape(-1)].reshape(B, L, -1).to(y.dtype)
        y = self.gamma * (y * n) + self.beta + y
        return h + self.pw2(y) * valid[..., None]


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        a = (t.float() * 1000)[:, None] * freqs[None]
        return self.mlp(torch.cat([torch.cos(a), torch.sin(a)], -1))


class Calibrator(nn.Module):
    """Input-dependent temperature tau(x) > 0 per question, from the mean slot representation and logit statistics.
    Initialised to tau = 1 (identity). Dividing logits by tau never changes the arg-max decision."""

    def __init__(self, d: int, hidden: int = 64):
        super().__init__()
        self.h = nn.Linear(d, hidden)
        self.f = nn.Linear(4, hidden)
        self.out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, math.log(math.exp(0.95) - 1.0))     # softplus(bias) + 0.05 = 1

    def forward(self, hq: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.out(F.silu(self.h(hq) + self.f(stats)))).squeeze(-1) + 0.05


class DecisionFlow(nn.Module):
    def __init__(self, backbone: str, dit_layers: int | None = None, convnext_blocks: int = 2, t_dim: int = 128,
                 score_encoding: str = "cdf", attn_dropout: float = 0.0, revision: str | None = None):
        super().__init__()
        enc = AutoModel.from_pretrained(backbone, revision=revision, dtype=torch.float32)
        self.backbone_name, self.backbone_revision = backbone, revision
        self.config = enc.config
        self.embeddings, self.layers, self.final_norm, self.rotary = enc.embeddings, enc.layers, enc.final_norm, enc.rotary_emb
        L = len(self.layers)
        self.n_dit = dit_layers if dit_layers is not None else max(2, round(0.3 * L))
        self.n_txt = L - self.n_dit
        d = self.config.hidden_size
        self.d = d
        self.window = self.config.sliding_window
        self.layer_types = list(self.config.layer_types)
        self.convnext = nn.ModuleList([ConvNeXtV2Block1d(d) for _ in range(convnext_blocks)])
        self.t_embed = TimestepEmbedder(t_dim)
        self.ans_in = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, d))
        nn.init.zeros_(self.ans_in[-1].weight); nn.init.zeros_(self.ans_in[-1].bias)
        self.mods = nn.ModuleList([nn.Linear(t_dim, 6 * d) for _ in range(self.n_dit)])
        for m in self.mods:
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        self.final_mod = nn.Linear(t_dim, 2 * d)
        nn.init.zeros_(self.final_mod.weight); nn.init.zeros_(self.final_mod.bias)
        self.head = nn.Linear(d, 1)
        nn.init.normal_(self.head.weight, std=0.02); nn.init.zeros_(self.head.bias)
        assert score_encoding in ("cdf", "onehot")
        self.score_encoding = score_encoding
        self.attn_dropout = attn_dropout
        self.calibrator = None

    def add_calibrator(self, hidden: int = 64):
        self.calibrator = Calibrator(self.d, hidden).to(self.head.weight.device)
        return self.calibrator

    def question_stats(self, z: torch.Tensor, batch: Batch) -> torch.Tensor:
        """(NQ, 4): top-1 minus top-2 logit, softmax entropy, log K, max probability."""
        NQ = len(batch.q_len)
        logp = seg_log_softmax(z, batch.slot_q, NQ)
        p = logp.exp()
        ent = -seg_sum(p * logp, batch.slot_q, NQ)
        zmax = torch.full((NQ,), -torch.inf, device=z.device).scatter_reduce(0, batch.slot_q, z, "amax", include_self=True)
        is_top = (z >= zmax[batch.slot_q])
        z2 = torch.where(is_top, torch.full_like(z, -torch.inf), z)
        zsec = torch.full((NQ,), -torch.inf, device=z.device).scatter_reduce(0, batch.slot_q, z2, "amax", include_self=True)
        margin = torch.where(torch.isfinite(zsec), zmax - zsec, torch.zeros_like(zmax))
        pmax = torch.full((NQ,), 0.0, device=z.device).scatter_reduce(0, batch.slot_q, p, "amax", include_self=True)
        return torch.stack([margin, ent, torch.log(batch.q_len.float()), pmax], -1)

    def calibrate(self, z: torch.Tensor, g_slots: torch.Tensor, batch: Batch) -> torch.Tensor:
        """Divide each question's logits by tau(x) if a calibrator is attached (arg-max preserving)."""
        if self.calibrator is None:
            return z
        NQ = len(batch.q_len)
        hq = torch.zeros(NQ, g_slots.shape[-1], device=z.device, dtype=torch.float32).index_add(0, batch.slot_q, g_slots.float())
        hq = hq / batch.q_len.float()[:, None]
        tau = self.calibrator(hq, self.question_stats(z.detach(), batch).detach())
        return z / tau[batch.slot_q]

    # -- parameters --------------------------------------------------------------------------------------------------
    def init_special_tokens(self, tokenizer):
        """Give the question / option markers the embeddings of [SEP] / [MASK] (they start as unused vocabulary)."""
        emb = self.embeddings.tok_embeddings.weight
        with torch.no_grad():
            emb[tokenizer.convert_tokens_to_ids(Q_TOKEN)] = emb[tokenizer.sep_token_id].clone()
            emb[tokenizer.convert_tokens_to_ids(OPT_TOKEN)] = emb[tokenizer.mask_token_id].clone()

    def param_groups(self, lr_backbone: float, lr_new: float, weight_decay: float):
        backbone = set(id(p) for m in (self.embeddings, self.layers, self.final_norm) for p in m.parameters())
        bb = [p for p in self.parameters() if id(p) in backbone and p.requires_grad]
        new = [p for p in self.parameters() if id(p) not in backbone and p.requires_grad]
        return [{"params": bb, "lr": lr_backbone, "weight_decay": weight_decay},
                {"params": new, "lr": lr_new, "weight_decay": weight_decay}]

    # -- forward pieces ----------------------------------------------------------------------------------------------
    def _rope(self, h, pos):
        return {lt: self.rotary(h, pos, lt) for lt in set(self.layer_types)}

    def text_branch(self, batch: Batch, nseg: int | None = None):
        """Everything that does not depend on the flow state: embeddings, the bottom encoder layers, ConvNeXt V2."""
        h = self.embeddings(input_ids=batch.ids)
        masks = build_masks(batch.seg, batch.pos, batch.valid, self.window)
        masks = {"full_attention": masks[0], "sliding_attention": masks[1]}
        rope = self._rope(h, batch.pos)
        drop = self.attn_dropout if self.training else 0.0
        for i in range(self.n_txt):
            layer = self.layers[i]
            lt = layer.attention_type
            h = h + attention(layer.attn, layer.attn_norm(h), masks[lt], rope[lt], drop)
            h = h + layer.mlp(layer.mlp_norm(h))
        B, L = batch.seg.shape
        nseg = nseg if nseg is not None else int(batch.seg.max().item()) + 2
        gid = (torch.arange(B, device=h.device)[:, None] * nseg + (batch.seg + 1)).long()
        for blk in self.convnext:
            h = blk(h, batch.seg, batch.valid, gid, B * nseg)
        return {"h": h, "masks": masks, "rope": rope}

    def flow_branch(self, cache, batch: Batch, x_t: torch.Tensor, c: torch.Tensor, t: torch.Tensor,
                    sc_flag: float = 0.0, return_hidden: bool = False) -> torch.Tensor:
        """DiT part: inject the answer track at the option slots, run the modulated layers, read option logits.
        x_t, c: (S,) flat slot values; t: (B,) flow time per request; sc_flag: 1 when c carries a self-condition /
        known-answer estimate, 0 when it is empty. Returns logits z: (S,)."""
        h = cache["h"]
        B, L, d = h.shape
        feats = torch.stack([x_t, c, torch.full_like(x_t, float(sc_flag))], -1).to(h.dtype)
        inj = self.ans_in(feats)
        h = h.index_put((batch.slot_b, batch.slot_i), inj, accumulate=True)
        temb = F.silu(self.t_embed(t)).to(h.dtype)                              # (B, t_dim)
        region = (batch.seg > 0).to(h.dtype)[..., None]                         # modulation only on question tokens
        drop = self.attn_dropout if self.training else 0.0
        for j in range(self.n_dit):
            layer = self.layers[self.n_txt + j]
            lt = layer.attention_type
            sa, ba, ga, sm, bm, gm = (m[:, None, :] * region for m in self.mods[j](temb).chunk(6, -1))
            x = layer.attn_norm(h) * (1 + sa) + ba
            h = h + (1 + ga) * attention(layer.attn, x, cache["masks"][lt], cache["rope"][lt], drop)
            x = layer.mlp_norm(h) * (1 + sm) + bm
            h = h + (1 + gm) * layer.mlp(x)
        fs, fb = (m[:, None, :] * region for m in self.final_mod(temb).chunk(2, -1))
        g = self.final_norm(h) * (1 + fs) + fb
        if return_hidden:
            return g
        g = g[batch.slot_b, batch.slot_i]                                       # (S, d)
        z = self.head(g).squeeze(-1).float()
        return self.calibrate(z, g, batch) if self.calibrator is not None else z

    # -- readout / encodings -----------------------------------------------------------------------------------------
    def encode_target(self, p: torch.Tensor, batch: Batch) -> torch.Tensor:
        """Distribution over options (S,) -> answer-track encoding: one-hot for yes/no & choice, CDF for score."""
        if self.score_encoding == "onehot" or not bool(batch.q_score.any()):
            return p
        cdf = seg_cumsum(p, batch.q_start, batch.slot_q)
        return torch.where(batch.q_score[batch.slot_q], cdf, p)

    def probs(self, z: torch.Tensor, batch: Batch) -> torch.Tensor:
        return seg_softmax(z, batch.slot_q, len(batch.q_len))

    def forward(self, batch: Batch, x_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None = None, cache=None):
        cache = cache if cache is not None else self.text_branch(batch)
        flag = 0.0 if c is None else 1.0
        c = c if c is not None else torch.zeros_like(x_t)
        z = self.flow_branch(cache, batch, x_t, c, t, sc_flag=flag)
        return z, cache

    # -- inference ---------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def readout(self, batch: Batch, nfe: int = 1, noise: str = "zero", generator=None, return_logits: bool = False):
        """System-1 readout (nfe=1: one pass at t=0 with the mean noise) or an Euler ODE with self-conditioning
        (nfe>1). Returns option probabilities (S,) (and logits of the last step)."""
        S = batch.slot_b.numel()
        dev = batch.ids.device
        x = torch.zeros(S, device=dev) if noise == "zero" else torch.randn(S, device=dev, generator=generator)
        c = None
        cache = self.text_branch(batch)
        B = batch.ids.shape[0]
        ts = torch.linspace(0, 1, nfe + 1, device=dev)
        z = None
        for n in range(nfe):
            t = ts[n].expand(B)
            z, _ = self.forward(batch, x, t, c, cache)
            p = self.probs(z, batch)
            x1 = self.encode_target(p, batch)
            c = x1
            if n < nfe - 1:
                x = x + (ts[n + 1] - ts[n]) * (x1 - x) / (1 - ts[n])
        p = self.probs(z, batch)
        return (p, z) if return_logits else p
