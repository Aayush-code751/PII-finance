"""Third-party validation corpora: adapters from raw HuggingFace schemas to the
canonical {id, text, gold:[Span]} shape the harness already consumes.

Each loader streams from HuggingFace (no full-dataset materialization), filters
to English / the relevant domain, maps native entity labels onto
CANONICAL_TYPES, and stops once `target_n` matching records are collected (or
the source is exhausted, whichever comes first -- some domain filters have
fewer than `target_n` rows available in total, which is reported, not padded).

Unmapped native labels are dropped, not force-fit onto the canonical taxonomy.
Every loader returns (records, stats) where stats reports how many native
spans were kept vs. dropped as unmapped, for honest disclosure in the paper.
"""
from __future__ import annotations
import random
from .core import Span

# ---- label maps: native entity label (lowercased) -> canonical type --------
GRETEL_LABEL_MAP = {
    "name": "PERSON",
    "date_of_birth": "DOB",
    "ssn": "SSN",
    "credit_card_number": "CREDIT_CARD",
    "iban": "IBAN",
    "phone_number": "PHONE",
    "email": "EMAIL",
    "street_address": "ADDRESS",
}

NEMOTRON_LABEL_MAP = {
    "first_name": "PERSON", "last_name": "PERSON", "full_name": "PERSON",
    "name": "PERSON", "human_name": "PERSON",
    "date_of_birth": "DOB",
    "street_address": "ADDRESS", "address": "ADDRESS",
    "email": "EMAIL", "email_address": "EMAIL",
    "phone_number": "PHONE", "phone": "PHONE",
    "ssn": "SSN", "social_security_number": "SSN",
    "credit_card_number": "CREDIT_CARD", "credit_card": "CREDIT_CARD",
    "iban": "IBAN",
    "account_number": "ACCOUNT_NUMBER", "bank_account_number": "ACCOUNT_NUMBER",
    "medical_record_number": "MRN", "mrn": "MRN",
    "insurance_id": "INSURANCE_ID", "policy_number": "INSURANCE_ID",
    "health_plan_id": "INSURANCE_ID", "insurance_policy_number": "INSURANCE_ID",
}

MEDDIES_LABEL_MAP = {
    "human_name": "PERSON",
    "date": "DOB",          # coarse: bundles all clinical dates, not just DOB
    "address": "ADDRESS",
    "email_address": "EMAIL",
    "phone_number": "PHONE",
    "id_number": "MRN",     # coarse: bundles MRN/SSN/beneficiary IDs
}


def _parse_native_spans(raw) -> list:
    """pii_spans/spans/label columns arrive as a list-of-dicts on some HF
    datasets, a JSON string on others, and a Python-repr string (single-quoted,
    as from str(list_of_dicts)) on nvidia/Nemotron-PII specifically. Normalize
    to list-of-dicts."""
    import ast as _ast
    import json as _json
    if raw is None:
        return []
    if isinstance(raw, str):
        parsed = None
        for parser in (_json.loads, _ast.literal_eval):
            try:
                parsed = parser(raw)
                break
            except (ValueError, TypeError, SyntaxError):
                continue
        if parsed is None:
            return []
        raw = parsed
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, str):
            try:
                parsed = _json.loads(item)
                if isinstance(parsed, dict):
                    out.append(parsed)
            except (ValueError, TypeError):
                continue
    return out


def _spans_from_native(text: str, native_spans, label_map: dict) -> tuple[list[Span], int, int]:
    kept, dropped = [], 0
    for sp in _parse_native_spans(native_spans):
        label = str(sp.get("label") or sp.get("category") or "").lower()
        canon = label_map.get(label)
        if canon is None:
            dropped += 1
            continue
        s, e = int(sp["start"]), int(sp["end"])
        txt = sp.get("text") or text[s:e]
        kept.append(Span(s, e, canon, txt))
    return kept, len(kept), dropped


def load_gretel_finance(target_n: int = 5000, seed: int = 42):
    """gretelai/synthetic_pii_finance_multilingual, English rows only.
    Apache-2.0. Native spans: {start,end,label} on `generated_text`."""
    import warnings
    warnings.filterwarnings("ignore")
    from datasets import load_dataset

    records, kept_spans, dropped_spans, scanned = [], 0, 0, 0
    for split in ("test", "train"):
        ds = load_dataset("gretelai/synthetic_pii_finance_multilingual",
                          split=split, streaming=True)
        for row in ds:
            scanned += 1
            if row.get("language") != "English":
                continue
            text = row.get("generated_text") or ""
            if not text:
                continue
            spans, k, d = _spans_from_native(text, row.get("pii_spans") or [],
                                             GRETEL_LABEL_MAP)
            kept_spans += k; dropped_spans += d
            records.append(dict(id=f"gretel-{split}-{row.get('index', len(records))}",
                                text=text, gold=spans, source="gretel_finance"))
            if len(records) >= target_n:
                break
        if len(records) >= target_n:
            break
    rng = random.Random(seed); rng.shuffle(records)
    stats = dict(source="gretelai/synthetic_pii_finance_multilingual",
                 license="Apache-2.0", scanned=scanned, kept_records=len(records),
                 target_n=target_n, kept_spans=kept_spans,
                 dropped_unmapped_spans=dropped_spans)
    return records[:target_n], stats


def load_nemotron(domain: str, target_n: int = 5000, seed: int = 42):
    """nvidia/Nemotron-PII filtered to a single `domain` value (e.g. "Finance",
    "Healthcare"). CC-BY-4.0. Native spans: {start,end,text,label} on `text`.
    Domain-filtered pools are small (~3-5k of 100k total); returns whatever is
    available if that is under target_n -- never padded."""
    import warnings
    warnings.filterwarnings("ignore")
    from datasets import load_dataset

    records, kept_spans, dropped_spans, scanned = [], 0, 0, 0
    ds = load_dataset("nvidia/Nemotron-PII", split="train", streaming=True)
    for row in ds:
        scanned += 1
        if row.get("domain") != domain:
            continue
        text = row.get("text") or ""
        if not text:
            continue
        spans, k, d = _spans_from_native(text, row.get("spans") or [],
                                         NEMOTRON_LABEL_MAP)
        kept_spans += k; dropped_spans += d
        records.append(dict(id=f"nemotron-{domain.lower()}-{row.get('uid', len(records))}",
                            text=text, gold=spans, source=f"nvidia_nemotron_{domain.lower()}"))
        if len(records) >= target_n:
            break
    rng = random.Random(seed); rng.shuffle(records)
    stats = dict(source=f"nvidia/Nemotron-PII[domain={domain}]", license="CC-BY-4.0",
                 scanned=scanned, kept_records=len(records), target_n=target_n,
                 kept_spans=kept_spans, dropped_unmapped_spans=dropped_spans)
    return records[:target_n], stats


def load_meddies(target_n: int = 5000, seed: int = 42):
    """Meddies/meddies-pii, `pii-bioes` config, English records only.
    CC-BY-NC-4.0 (attribution, non-commercial). Native spans:
    {category,start,end,text} on `text`."""
    import warnings
    warnings.filterwarnings("ignore")
    from datasets import load_dataset

    records, kept_spans, dropped_spans, scanned = [], 0, 0, 0
    ds = load_dataset("Meddies/meddies-pii", "pii-bioes", split="train", streaming=True)
    for row in ds:
        scanned += 1
        info = row.get("info") or {}
        if info.get("language") not in ("English", "en"):
            continue
        text = row.get("text") or ""
        if not text:
            continue
        native = [dict(label=s.get("category"), start=s["start"], end=s["end"],
                       text=s.get("text")) for s in (row.get("label") or [])]
        spans, k, d = _spans_from_native(text, native, MEDDIES_LABEL_MAP)
        kept_spans += k; dropped_spans += d
        records.append(dict(id=f"meddies-{info.get('id', len(records))}",
                            text=text, gold=spans, source="meddies_pii"))
        if len(records) >= target_n:
            break
    rng = random.Random(seed); rng.shuffle(records)
    stats = dict(source="Meddies/meddies-pii[pii-bioes,en]",
                 license="CC-BY-NC-4.0 (attribution, non-commercial)",
                 scanned=scanned, kept_records=len(records), target_n=target_n,
                 kept_spans=kept_spans, dropped_unmapped_spans=dropped_spans)
    return records[:target_n], stats
