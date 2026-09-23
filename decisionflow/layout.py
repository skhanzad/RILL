"""Token layout for a decision request.

A request is one state and Q typed questions (rendered to text by kev.api, so the text the model sees is the text any
System One client sends). The packed sequence is

    [CLS] state [SEP] | [Q] instructions [OPT] option_1 [OPT] option_2 ... [SEP] | [Q] ... [SEP] | ...

Segment 0 is the state; segment j (1-based) is question j. Positions restart after the state for every question, and
the attention masks (model.build_masks) let a question read the state and itself only, so every question is answered
exactly as if it had been asked alone, and the state encoding never depends on the questions (it can be cached).

Each [OPT] token is an *answer slot*: the flow-matching target lives on these positions only (one scalar per option).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

Q_TOKEN = "[unused0]"
OPT_TOKEN = "[unused1]"

MAX_OPTION_TOKENS = 64
MAX_INSTR_TOKENS = 256


@dataclass
class Encoded:
    ids: np.ndarray            # int32 token ids
    seg: np.ndarray            # int16 segment of each token (0 = state, j = question j)
    pos: np.ndarray            # int32 position ids
    slots: list[list[int]]            # per question: token indices of its [OPT] markers
    qtypes: list[str]
    targets: list[list[float] | None] = field(default_factory=list)   # per question: distribution over options, if labelled
    labels: list[int | None] = field(default_factory=list)


class Layout:
    def __init__(self, tokenizer, max_state: int | None = 512, max_total: int | None = None):
        self.tok = tokenizer
        self.cls, self.sep, self.pad = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
        self.q_id = tokenizer.convert_tokens_to_ids(Q_TOKEN)
        self.opt_id = tokenizer.convert_tokens_to_ids(OPT_TOKEN)
        assert self.q_id != tokenizer.unk_token_id and self.opt_id != tokenizer.unk_token_id
        self.max_state, self.max_total = max_state, max_total

    def encode(self, rec: dict) -> Encoded:
        """rec: an internal record from kev.data.materialize (labelled) or kev.api.to_record (serving):
        {"state": str, "questions": [{"instr": str, "options": [str], "qtype"?: str, "label"?: int, "target"?: [float]}]}"""
        pieces = [rec["state"] or ""]
        for q in rec["questions"]:
            pieces.append(q.get("instr") or "")
            pieces += [" " + o for o in q["options"]]
        toks = self.tok(pieces, add_special_tokens=False)["input_ids"]
        st = toks[0][:self.max_state] if self.max_state else toks[0]
        ids = [self.cls] + list(st) + [self.sep]
        S = len(ids)
        seg, pos = [0] * S, list(range(S))
        slots, qtypes, targets, labels = [], [], [], []
        k = 1
        for j, q in enumerate(rec["questions"], start=1):
            qids = [self.q_id] + list(toks[k][:MAX_INSTR_TOKENS]); k += 1
            qslots = []
            for _ in q["options"]:
                qslots.append(len(ids) + len(qids))
                qids += [self.opt_id] + list(toks[k][:MAX_OPTION_TOKENS]); k += 1
            qids.append(self.sep)
            ids += qids
            seg += [j] * len(qids)
            pos += list(range(S, S + len(qids)))
            slots.append(qslots)
            qtypes.append(q.get("qtype", "choice"))
            K = len(q["options"])
            if q.get("target") is not None:
                targets.append([float(x) for x in q["target"]]); labels.append(None)
            elif q.get("label") is not None and "keys" in q:      # labelled training/eval record
                y = int(q["label"]); t = [0.0] * K; t[y] = 1.0
                targets.append(t); labels.append(y)
            else:
                targets.append(None); labels.append(None)
        if self.max_total and len(ids) > self.max_total:
            raise ValueError(f"packed request has {len(ids)} tokens > {self.max_total}")
        return Encoded(np.asarray(ids, np.int32), np.asarray(seg, np.int16), np.asarray(pos, np.int32), slots, qtypes, targets, labels)


@dataclass
class Batch:
    ids: torch.Tensor          # (B, L)
    seg: torch.Tensor          # (B, L)  -1 on padding
    pos: torch.Tensor          # (B, L)
    valid: torch.Tensor        # (B, L) bool
    slot_b: torch.Tensor       # (S,) batch index of each answer slot
    slot_i: torch.Tensor       # (S,) token index of each answer slot
    slot_q: torch.Tensor       # (S,) global question index of each slot (questions numbered across the batch)
    q_start: torch.Tensor      # (NQ,) first flat slot of each question
    q_len: torch.Tensor        # (NQ,) number of options
    q_b: torch.Tensor          # (NQ,) batch index of each question
    q_score: torch.Tensor      # (NQ,) bool: ordinal (score) question
    target: torch.Tensor | None   # (S,) target distribution per slot (concatenated per question), or None
    has_target: torch.Tensor | None  # (NQ,) bool
    q_type: list[str]

    def to(self, device):
        out = {}
        for k, v in self.__dict__.items():
            out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        return Batch(**out)


def collate(encs: list[Encoded], pad_id: int) -> Batch:
    B = len(encs)
    L = max(len(e.ids) for e in encs)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    seg = torch.full((B, L), -1, dtype=torch.long)
    pos = torch.zeros((B, L), dtype=torch.long)
    valid = torch.zeros((B, L), dtype=torch.bool)
    slot_b, slot_i, slot_q, q_start, q_len, q_b, q_score, target, has_t, q_type = [], [], [], [], [], [], [], [], [], []
    nq = 0
    for b, e in enumerate(encs):
        n = len(e.ids)
        ids[b, :n] = torch.from_numpy(np.asarray(e.ids, np.int64)); seg[b, :n] = torch.from_numpy(np.asarray(e.seg, np.int64))
        pos[b, :n] = torch.from_numpy(np.asarray(e.pos, np.int64)); valid[b, :n] = True
        for j, s in enumerate(e.slots):
            q_start.append(len(slot_b)); q_len.append(len(s)); q_b.append(b); q_score.append(e.qtypes[j] == "score"); q_type.append(e.qtypes[j])
            slot_b += [b] * len(s); slot_i += s; slot_q += [nq] * len(s)
            t = e.targets[j] if e.targets else None
            if t is None:
                target += [0.0] * len(s); has_t.append(False)
            else:
                target += list(t); has_t.append(True)
            nq += 1
    as_long = lambda x: torch.tensor(x, dtype=torch.long)
    any_t = any(has_t)
    return Batch(ids, seg, pos, valid, as_long(slot_b), as_long(slot_i), as_long(slot_q), as_long(q_start), as_long(q_len),
                 as_long(q_b), torch.tensor(q_score, dtype=torch.bool),
                 torch.tensor(target, dtype=torch.float32) if any_t else None,
                 torch.tensor(has_t, dtype=torch.bool) if any_t else None, q_type)
