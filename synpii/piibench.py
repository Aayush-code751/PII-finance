"""PIIBench (Pritesh-2711/pii-bench) adapter.

* Robust BIO-token -> character-span alignment (wordpiece '##', [UNK],
  case-folding and diacritic drift between tokens and source text).
* Label maps: PIIBench gold -> canonical 11-type taxonomy (evaluation of the
  paper's framework), and canonical predictions -> PIIBench label space
  (seqeval protocol identical to run_existing_models_benchmark.py upstream).
* Official comparative-subset protocol: source-stratified largest-remainder
  sampling, seed 42, ported from PIIBench's create_evaluation_subset.py.
"""
from __future__ import annotations
import json, random, unicodedata
from collections import defaultdict
from .core import Span

# ---- PIIBench gold label -> canonical 11-type taxonomy ---------------------
PIIB_TO_CANON = {
    "PERSON": "PERSON", "NAME": "PERSON",
    "EMAIL": "EMAIL",
    "PHONE_NUMBER": "PHONE", "PHONE": "PHONE", "TELEPHONENUM": "PHONE",
    "SSN": "SSN",
    "CREDIT_CARD": "CREDIT_CARD", "CREDIT_CARD_NUMBER": "CREDIT_CARD",
    "CREDIT_DEBIT_CARD": "CREDIT_CARD",
    "IBAN": "IBAN", "IBAN_CODE": "IBAN",
    "ACCOUNT_NUMBER": "ACCOUNT_NUMBER",
    "DATE_OF_BIRTH": "DOB",
    "ADDRESS": "ADDRESS", "STREET_ADDRESS": "ADDRESS",
    "MEDICAL_RECORD_NUMBER": "MRN", "MRN": "MRN",
    "INSURANCE_ID": "INSURANCE_ID", "HEALTH_PLAN_ID": "INSURANCE_ID",
}
# canonical prediction -> PIIBench label space (for the seqeval protocol)
CANON_TO_PIIB = {
    "PERSON": "PERSON", "EMAIL": "EMAIL", "PHONE": "PHONE_NUMBER", "SSN": "SSN",
    "CREDIT_CARD": "CREDIT_CARD", "IBAN": "IBAN", "ACCOUNT_NUMBER": "ACCOUNT_NUMBER",
    "DOB": "DATE_OF_BIRTH", "ADDRESS": "ADDRESS", "MRN": None, "INSURANCE_ID": None,
}

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s.casefold() if not unicodedata.combining(c))

def align_tokens(tokens: list[str], text: str):
    """Return per-token (start,end) char offsets in `text`, or None if a token
    cannot be located. Greedy left-to-right with small skip tolerance."""
    # map: normalized index -> original index
    n_chars, idx_map = [], []
    for i, ch in enumerate(text):
        for nc in _norm(ch):
            n_chars.append(nc); idx_map.append(i)
    ntext = "".join(n_chars)
    offsets, cursor = [], 0
    for tok in tokens:
        piece = tok[2:] if tok.startswith("##") else tok
        if piece == "[UNK]" or piece == "":
            offsets.append(None); continue
        npiece = _norm(piece)
        if not npiece:
            offsets.append(None); continue
        j = ntext.find(npiece, cursor)
        if j == -1 or j - cursor > 24:          # lost alignment for this token
            j2 = ntext.find(npiece, cursor)
            if j2 == -1:
                offsets.append(None); continue
            j = j2
        s_orig = idx_map[j]
        e_orig = idx_map[min(j + len(npiece) - 1, len(idx_map) - 1)] + 1
        offsets.append((s_orig, e_orig))
        cursor = j + len(npiece)
    return offsets

def bio_to_spans(tokens, labels, text) -> tuple[list[Span], bool]:
    """PIIBench gold BIO -> char spans on `text`. seqeval span-start
    convention (B-, or I- after O / type change, matching upstream)."""
    offs = align_tokens(tokens, text)
    spans, ok = [], True
    cur_type, cur_s, cur_e = None, None, None
    def flush():
        nonlocal cur_type, cur_s, cur_e
        if cur_type is not None and cur_s is not None:
            spans.append(Span(cur_s, cur_e, cur_type, text[cur_s:cur_e]))
        cur_type = cur_s = cur_e = None
    prev = "O"
    for tok, lab, off in zip(tokens, labels, offs):
        if lab == "O":
            flush(); prev = "O"; continue
        pre, typ = lab.split("-", 1)
        starts = pre == "B" or prev == "O" or (cur_type != typ)
        if starts:
            flush(); cur_type = typ
            if off is None:
                ok = False
            else:
                cur_s, cur_e = off
        else:
            if off is not None:
                if cur_s is None: cur_s = off[0]
                cur_e = off[1]
            else:
                ok = False
        prev = pre
    flush()
    return spans, ok

def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: out.append(json.loads(line))
    return out

def prepare_records(records: list[dict], canon_only: bool = False):
    """-> list of {text, source, gold_piib:[Span], gold_canon:[Span], aligned}"""
    out, n_bad = [], 0
    for r in records:
        text = r.get("text") or " ".join(r["tokens"])
        spans, ok = bio_to_spans(r["tokens"], r["labels"], text)
        if not ok: n_bad += 1
        canon = [Span(s.start, s.end, PIIB_TO_CANON[s.type], s.text)
                 for s in spans if s.type in PIIB_TO_CANON]
        out.append(dict(text=text, source=r.get("source", "?"),
                        gold_piib=spans, gold_canon=canon, aligned=ok))
    return out, n_bad

# ---- Official comparative subset protocol (ported from upstream) -----------
def make_stratified_subset(records: list[dict], target_size: int, seed: int = 42):
    if target_size >= len(records): return list(records)
    rng = random.Random(seed)
    by_source = defaultdict(list)
    for rec in records: by_source[rec["source"]].append(rec)
    total = len(records)
    quota = {s: target_size * len(v) / total for s, v in by_source.items()}
    alloc = {s: int(q) for s, q in quota.items()}
    rem = target_size - sum(alloc.values())
    for s, _ in sorted(quota.items(), key=lambda kv: -(kv[1] - int(kv[1])))[:rem]:
        alloc[s] += 1
    subset = []
    for s, recs in by_source.items():
        subset.extend(rng.sample(recs, min(alloc[s], len(recs))))
    rng.shuffle(subset)
    return subset

# ---- seqeval-protocol scoring on the PIIBench label space ------------------
def spans_to_bio_tokens(tokens, text, pred: list[Span], to_piib: bool = True):
    offs = align_tokens(tokens, text)
    bio = ["O"] * len(tokens)
    for sp in pred:
        typ = CANON_TO_PIIB.get(sp.type, sp.type) if to_piib else sp.type
        if typ is None: continue
        first = True
        for i, off in enumerate(offs):
            if off is None: continue
            s, e = off
            if s < sp.end and e > sp.start:          # token overlaps span
                bio[i] = ("B-" if first else "I-") + typ
                first = False
    return bio
