"""Baseline predictors, all emitting kev-benchmark predictions ({"probabilities": {qid: {key: p}}, "logits": ...}) so every
system is scored by the same kev.metrics code.

  SemIfDirect   SemIf's direct option-letter readout (its frozen system prompt + JSON payload, native chat template,
                last-position logits restricted to the answer letters). Used for SemIf's own models (Qwen3.5-4B,
                Qwen3-0.6B = the JEV-CPU configuration, MiniCPM5-2B) and, with the same protocol, for the Layla models.
  Generative    The same prompt, but the model *generates* its answer (greedy, <= 8 tokens) and the answer is parsed.
                Unparseable outputs are "invalid" (a format hallucination). Probabilities: one-hot on the parsed answer
                (uniform when invalid), which is how a chat model's free-text decision is consumed in practice.
  NLIZeroShot   Entailment-based zero-shot classifiers (Yin et al., 2019): one premise/hypothesis pass per option.
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "third_party" / "kev", ROOT / "third_party" / "semif" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from kev.api import option_text, question_keys, render  # noqa: E402
from semif_phase1.core import DIRECT_SYSTEM, LETTERS, direct_messages, load_causal_model  # noqa: E402
from semif_phase1.direct import _forward, encode_prompt  # noqa: E402


class Unsupported(Exception):
    """The system cannot represent this question (e.g. more options than its answer alphabet)."""


def question_options(q: dict) -> tuple[list[str], list[str]]:
    """(keys, option texts) of a request question, rendered exactly as kev.api.to_record renders them."""
    keys = question_keys(q["type"], q.get("criteria"))
    if q["type"] == "noul":
        c = q.get("criteria") or {}
        texts = [option_text("no", c.get("false")), option_text("yes", c.get("true"))]
    elif q["type"] == "choice":
        texts = [option_text(k, v) for k, v in q["criteria"].items()]
    else:
        texts = [render(x) for x in q["criteria"]]
    return keys, texts


def question_text(q: dict) -> str:
    text = render(q.get("instructions")).strip()
    if text:
        return text
    return {"noul": "Is the statement true?", "choice": "Which option applies?", "score": "Which level applies?"}[q["type"]]


def semif_row(record: dict, qid: str, q: dict) -> dict:
    keys, texts = question_options(q)
    if len(keys) > len(LETTERS):
        raise Unsupported(f"{len(keys)} options > {len(LETTERS)} answer letters")
    return {"id": f"{record['_meta']['id']}::{qid}", "state": record["state"] if record["state"] not in ("", None, [], {}) else " ",
            "question": question_text(q), "options": [{"id": k, "description": t} for k, t in zip(keys, texts)]}


def _softmax(xs):
    m = max(xs); e = [math.exp(x - m) for x in xs]; s = sum(e)
    return [v / s for v in e]


class LetterScorer:
    """SemIf's direct protocol. For tokenizers where SemIf's strict single-token boundary check fails (sentencepiece
    Llama/Mistral vocabularies put a space marker on the letter), the answer token is taken as the first token the model
    would emit for the letter after the generation prompt; the number of such rows is reported."""

    def __init__(self, model_id: str, revision: str | None, dtype: str = "bfloat16", chat_template: str | None = None,
                 quant: str | None = None, max_tokens: int = 8192, letter_prefix: str = ""):
        self.model_id, self.revision, self.max_tokens = model_id, revision, max_tokens
        self.letter_prefix = letter_prefix
        if quant is None and revision and re.fullmatch(r"[0-9a-f]{40}", revision):
            self.model, self.tok, self.meta = load_causal_model(model_id, revision, "cuda", dtype)
        else:
            import transformers
            kw = {"dtype": getattr(torch, dtype), "device_map": {"": "cuda:0"}}
            if quant == "nf4":
                kw["quantization_config"] = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                                            bnb_4bit_compute_dtype=torch.bfloat16)
                kw.pop("dtype")
            elif quant == "int8":
                kw["quantization_config"] = transformers.BitsAndBytesConfig(load_in_8bit=True); kw.pop("dtype")
            self.tok = transformers.AutoTokenizer.from_pretrained(model_id, revision=revision)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(model_id, revision=revision, **kw).eval()
            self.meta = {"source": model_id, "revision": revision, "dtype": dtype, "quant": quant}
        if chat_template is not None:
            self.tok.chat_template = chat_template
        self.fallback_rows = 0
        self.temperature = 1.0

    def prompt_ids(self, row: dict) -> tuple[list[int], list[int]]:
        ids, slots = self._prompt_ids(row)
        # SemIf encodes without special tokens, which is right for its Qwen/MiniCPM templates (MiniCPM's template emits
        # BOS itself); Llama/Mistral-family tokenizers expect a leading <s>, so add it when the tokenizer adds one by default.
        bos = getattr(self.tok, "bos_token_id", None)
        if getattr(self.tok, "add_bos_token", False) and bos is not None and (not ids or ids[0] != bos):
            ids = [bos] + ids
        return ids, slots

    def _prompt_ids(self, row: dict) -> tuple[list[int], list[int]]:
        try:
            if self.letter_prefix:
                raise ValueError("letter prefix protocol")
            ids, slots, _ = encode_prompt(self.tok, row, self.max_tokens)
            return ids, slots
        except ValueError as e:
            if "exceed limit" in str(e):
                raise Unsupported(str(e))
            # tokenizer-robust fallback (same prompt text, same letters)
            prompt = self.tok.apply_chat_template(direct_messages(row), tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
            ids = self.tok.encode(prompt, add_special_tokens=False)
            if len(ids) > self.max_tokens:
                raise Unsupported("prompt too long")
            slots = []
            for letter in LETTERS[:len(row["options"])]:
                full = self.tok.encode(prompt + self.letter_prefix + letter, add_special_tokens=False)
                extra = full[len(ids):] if full[:len(ids)] == ids else self.tok.encode(self.letter_prefix + letter, add_special_tokens=False)
                extra = [t for t in extra if self.tok.decode([t]).strip()]   # drop a bare space-marker token, keep the letter
                slots.append(extra[0])
            if len(set(slots)) != len(slots):
                raise Unsupported("answer letters collide in this vocabulary")
            self.fallback_rows += 1
            return ids, slots

    @torch.no_grad()
    def score_row(self, row: dict) -> list[float]:
        ids, slots = self.prompt_ids(row)
        dev = next(self.model.parameters()).device
        inputs = {"input_ids": torch.tensor([ids], device=dev), "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=dev)}
        logits = _forward(self.model, inputs)[0].float()
        return logits[slots].cpu().tolist()

    def __call__(self, record: dict) -> dict:
        probs, logits = {}, {}
        t0 = time.perf_counter()
        n_tok = 0
        for qid, q in record["questions"].items():
            row = semif_row(record, qid, q)
            z = self.score_row(row)
            keys = [o["id"] for o in row["options"]]
            probs[qid] = dict(zip(keys, _softmax(z))); logits[qid] = dict(zip(keys, z))
        if torch.cuda.is_available(): torch.cuda.synchronize()
        return {"probabilities": probs, "logits": logits, "inference_temperature": 1.0,
                "latency_ms": 1000 * (time.perf_counter() - t0), "input_tokens": n_tok}


ANSWER_RE = re.compile(r"\b([A-P])\b")


class Generative(LetterScorer):
    """Free-text use of a chat model: same prompt, greedy generation, parse the first answer letter."""

    def __init__(self, *a, max_new_tokens: int = 8, **k):
        super().__init__(*a, **k)
        self.max_new_tokens = max_new_tokens
        self.invalid = 0
        self.total = 0
        self.outputs = []

    @torch.no_grad()
    def generate_row(self, row):
        ids, _ = self.prompt_ids(row)
        dev = next(self.model.parameters()).device
        x = torch.tensor([ids], device=dev)
        out = self.model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=self.max_new_tokens, do_sample=False,
                                  pad_token_id=self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id)
        return self.tok.decode(out[0, len(ids):], skip_special_tokens=True)

    def __call__(self, record):
        probs = {}
        t0 = time.perf_counter()
        for qid, q in record["questions"].items():
            row = semif_row(record, qid, q)
            text = self.generate_row(row)
            K = len(row["options"])
            m = ANSWER_RE.search(text.strip().upper()[:12]) if text else None
            letter_ok = m is not None and LETTERS.index(m.group(1)) < K
            self.total += 1
            if letter_ok:
                y = LETTERS.index(m.group(1))
                p = [0.0] * K; p[y] = 1.0
            else:
                self.invalid += 1
                p = [1.0 / K] * K
            self.outputs.append({"id": row["id"], "text": text, "valid": letter_ok})
            probs[qid] = dict(zip([o["id"] for o in row["options"]], p))
        return {"probabilities": probs, "latency_ms": 1000 * (time.perf_counter() - t0), "input_tokens": 0}


class NLIZeroShot:
    """premise = rendered state; hypothesis = 'Question: <instructions> Answer: <option>.' ; one pass per option.
    Two-class models (entailment / not_entailment) use logit(entail) - logit(not); three-class MNLI models use the
    entailment logit; the option scores are softmaxed across options (single-label zero-shot classification)."""

    def __init__(self, model_id: str, batch: int = 64, max_length: int = 512):
        import transformers
        self.tok = transformers.AutoTokenizer.from_pretrained(model_id)
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(model_id, dtype=torch.float16).cuda().eval()
        lab = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.ent = next(v for k, v in lab.items() if k.startswith("entail"))
        self.neg = lab.get("not_entailment", lab.get("contradiction"))
        self.two_class = "not_entailment" in lab
        self.batch, self.max_length = batch, max_length
        self.temperature = 1.0

    @torch.no_grad()
    def __call__(self, record):
        premise = render(record["state"])
        probs, logits = {}, {}
        t0 = time.perf_counter()
        for qid, q in record["questions"].items():
            keys, texts = question_options(q)
            instr = question_text(q)
            hyps = [f"Question: {instr} Answer: {t}." for t in texts]
            zs = []
            for i in range(0, len(hyps), self.batch):
                enc = self.tok([premise] * len(hyps[i:i + self.batch]), hyps[i:i + self.batch], truncation="only_first",
                               max_length=self.max_length, padding=True, return_tensors="pt").to("cuda")
                out = self.model(**enc).logits.float()
                s = out[:, self.ent] - out[:, self.neg] if self.two_class else out[:, self.ent]
                zs += s.cpu().tolist()
            probs[qid] = dict(zip(keys, _softmax(zs))); logits[qid] = dict(zip(keys, zs))
        if torch.cuda.is_available(): torch.cuda.synchronize()
        return {"probabilities": probs, "logits": logits, "inference_temperature": 1.0,
                "latency_ms": 1000 * (time.perf_counter() - t0), "input_tokens": 0}
