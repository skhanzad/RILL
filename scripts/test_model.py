"""Unit checks for DecisionFlow: identity-at-init vs the pretrained encoder, question isolation, segment ops, backward."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "third_party/kev"))
import torch
from transformers import AutoModel, AutoTokenizer
from decisionflow.layout import Layout, collate
from decisionflow.model import DecisionFlow, seg_softmax, seg_cumsum, build_masks

torch.manual_seed(0)
dev = "cuda"
name = sys.argv[1] if len(sys.argv) > 1 else "jhu-clsp/ettin-encoder-68m"
tok = AutoTokenizer.from_pretrained(name)
lay = Layout(tok)
model = DecisionFlow(name).to(dev).eval()
model.init_special_tokens(tok)
torch.backends.cuda.matmul.allow_tf32 = False

# 1) identity at init: a state-only sequence through our text+flow branches == HF encoder output
text = "The customer was charged twice for order 4411 and wants a refund of one charge. " * 12
enc = lay.encode({"state": text, "questions": []})
b = collate([enc], tok.pad_token_id).to(dev)
hf = AutoModel.from_pretrained(name, dtype=torch.float32, attn_implementation="sdpa").to(dev).eval()
with torch.no_grad():
    ref = hf(input_ids=b.ids).last_hidden_state
    cache = model.text_branch(b)
    g = model.flow_branch(cache, b, torch.zeros(0, device=dev), torch.zeros(0, device=dev), torch.zeros(1, device=dev), return_hidden=True)
print("identity@init max|diff| =", (g - ref).abs().max().item(), " seq len", b.ids.shape[1])

# 2) isolation: question 2 answered in a packed request == asked alone
rec = {"state": text, "questions": [
    {"instr": "Which team should handle this?", "options": ["billing: charges and refunds", "shipping: delivery", "returns"], "qtype": "choice"},
    {"instr": "Is the customer angry?", "options": ["no", "yes"], "qtype": "noul"},
    {"instr": "How urgent is it?", "options": ["can wait", "this week", "today"], "qtype": "score"}]}
# perturb the flow parameters so that isolation is tested on a non-trivial network
with torch.no_grad():
    for p in list(model.mods.parameters()) + list(model.ans_in.parameters()) + list(model.final_mod.parameters()):
        p.add_(0.02 * torch.randn_like(p))
    for blk in model.convnext:
        blk.pw2.weight.add_(0.02 * torch.randn_like(blk.pw2.weight))
packed = collate([lay.encode(rec)], tok.pad_token_id).to(dev)
alone = [collate([lay.encode({"state": text, "questions": [q]})], tok.pad_token_id).to(dev) for q in rec["questions"]]
with torch.no_grad():
    S = packed.slot_b.numel()
    xt = torch.randn(S, device=dev)
    t = torch.full((1,), 0.37, device=dev)
    zp, _ = model(packed, xt, t)
    zs = []
    off = 0
    for j, a in enumerate(alone):
        k = a.slot_b.numel()
        z, _ = model(a, xt[off:off + k], t)
        zs.append(z); off += k
    zs = torch.cat(zs)
print("isolation max|diff| =", (zp - zs).abs().max().item(), "logits", zp.tolist())

# 3) batched vs single (padding invariance)
r2 = {"state": "Short state.", "questions": [rec["questions"][1]]}
bb = collate([lay.encode(rec), lay.encode(r2)], tok.pad_token_id).to(dev)
with torch.no_grad():
    S1 = packed.slot_b.numel()
    x2 = torch.randn(2, device=dev)
    zb, _ = model(bb, torch.cat([xt, x2]), torch.tensor([0.37, 0.37], device=dev))
    z2, _ = model(collate([lay.encode(r2)], tok.pad_token_id).to(dev), x2, t)
print("padding max|diff| =", max((zb[:S1] - zp).abs().max().item(), (zb[S1:] - z2).abs().max().item()))

# 4) segment ops
z = torch.tensor([1., 2., 3., 0., 1., 5., 5., 5.], device=dev)
seg = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], device=dev)
p = seg_softmax(z, seg, 3)
print("seg_softmax sums:", torch.zeros(3, device=dev).index_add(0, seg, p).tolist())
print("seg_cumsum:", seg_cumsum(p, torch.tensor([0, 3, 5], device=dev), seg).tolist())

# 5) backward
model.train()
with torch.autocast("cuda", dtype=torch.bfloat16):
    z, _ = model(bb, torch.randn(bb.slot_b.numel(), device=dev), torch.rand(2, device=dev))
    loss = z.pow(2).mean()
loss.backward()
ng = sum(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters())
print("params with grad:", ng, "/", sum(1 for _ in model.parameters()))
print("n_txt", model.n_txt, "n_dit", model.n_dit, "params(M)", sum(p.numel() for p in model.parameters()) / 1e6)
missing = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum().item() == 0]
print("no-grad params:", len(missing)); print(missing[:40])
nan = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
print("non-finite grads:", len(nan), nan[:10])
