"""Detection engines behind a single interface: detect(text) -> list[Span]."""
from __future__ import annotations
import json, re, urllib.request
from .core import Span, luhn_valid, iban_valid

# ---------------------------------------------------------------------------
# Rule engine: checksummed surface patterns + context-window evidence.
# Context vocabulary is generic domain vocabulary (account, patient, DOB...),
# deliberately NOT extended with phrases from our own synthetic templates.
# ---------------------------------------------------------------------------
CTX = 48  # context window, chars each side

def _ctx(text, s, e):
    return text[max(0, s - CTX):min(len(text), e + CTX)].lower()

class RuleEngine:
    name = "rule"

    def detect(self, text: str) -> list[Span]:
        out: list[Span] = []
        add = lambda s, e, t, sc: out.append(Span(s, e, t, text[s:e], sc))

        for m in re.finditer(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
            add(m.start(), m.end(), "EMAIL", 0.97)

        for m in re.finditer(r"\b(?:\d[ -]?){12,18}\d\b", text):
            digits = re.sub(r"\D", "", m.group())
            if 13 <= len(digits) <= 19:
                add(m.start(), m.end(), "CREDIT_CARD",
                    0.98 if luhn_valid(digits) else 0.35)

        for m in re.finditer(r"\b[A-Z]{2}\d{2}[ ]?[A-Z0-9][A-Z0-9 ]{8,32}[A-Z0-9]\b", text):
            add(m.start(), m.end(), "IBAN", 0.99 if iban_valid(m.group()) else 0.40)

        for m in re.finditer(r"\b\d{3}-\d{2}-\d{4}\b", text):
            a = int(m.group()[:3])
            ok = a not in (0, 666) and a < 900
            add(m.start(), m.end(), "SSN", 0.96 if ok else 0.50)
        for m in re.finditer(r"\b\d{9}\b", text):
            if re.search(r"ssn|social security", _ctx(text, m.start(), m.end())):
                add(m.start(), m.end(), "SSN", 0.85)

        for m in re.finditer(
            r"(?:\+?1[ .-]?)?(?:\(\d{3}\)[ .-]?|\d{3}[ .-])\d{3}[ .-]\d{4}\b", text):
            c = _ctx(text, m.start(), m.end())
            sc = 0.92 if re.search(r"phone|tel|call|fax|mobile|contact|cell", c) else 0.55
            add(m.start(), m.end(), "PHONE", sc)

        for m in re.finditer(
            r"\b(?:\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2}|(?:19|20)\d{2}-\d{2}-\d{2}|"
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? (?:19|20)\d{2})\b",
            text):
            c = _ctx(text, m.start(), m.end())
            sc = 0.90 if re.search(r"\bdob\b|birth|born|geb", c) else 0.20
            add(m.start(), m.end(), "DOB", sc)

        for m in re.finditer(r"\b\d{8,17}\b", text):
            c = _ctx(text, m.start(), m.end())
            if re.search(r"account|acct|a/c|routing|wire", c):
                add(m.start(), m.end(), "ACCOUNT_NUMBER", 0.88)

        for m in re.finditer(
            r"\b\d{1,5}\s+[A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+)?\s"
            r"(?:St|Street|Ave|Avenue|Rd|Road|Ln|Lane|Blvd|Boulevard|Dr|Drive|Ct|Court|Way)\b"
            r"(?:,?\s[A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+)?,?\s[A-Z]{2}\s\d{5})?", text):
            sc = 0.92 if re.search(r"\d{5}$", m.group()) else 0.80
            add(m.start(), m.end(), "ADDRESS", sc)

        for m in re.finditer(r"(?:MRN|[Mm]edical [Rr]ecord(?: [Nn]o\.?| [Nn]umber)?)[:# ]*([A-Z0-9-]{5,12})", text):
            add(m.start(1), m.end(1), "MRN", 0.93)

        for m in re.finditer(
            r"(?:[Ii]nsurance|[Pp]olicy|[Mm]ember|[Pp]lan)\s*(?:ID|[Nn]o\.?|[Nn]umber)?\s*[:#]\s*([A-Z0-9-]{6,15})",
            text):
            add(m.start(1), m.end(1), "INSURANCE_ID", 0.87)

        # PERSON: honorific / role-anchored capitalized sequences, plus a weak
        # bare capitalized-bigram heuristic.
        namepat = r"([A-Z][a-z]+(?:\s[A-Z]\.)?(?:\s[A-Z][a-z]+){1,2})"
        for m in re.finditer(r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s" + namepat, text):
            add(m.start(1), m.end(1), "PERSON", 0.85)
        for m in re.finditer(
            r"(?:patient|borrower|applicant|client|customer|beneficiary|physician|guarantor|holder|contact)"
            r"\s*[,:]?\s+" + namepat, text, re.I):
            add(m.start(1), m.end(1), "PERSON", 0.78)
        for m in re.finditer(r"\b([A-Z][a-z]{2,})\s([A-Z][a-z]{2,})\b", text):
            add(m.start(), m.end(), "PERSON", 0.40)

        return _dedupe_overlaps(out)

def _dedupe_overlaps(spans: list[Span]) -> list[Span]:
    """Keep highest-score span among same-type heavy overlaps."""
    spans = sorted(spans, key=lambda s: (-s.score, s.start))
    kept: list[Span] = []
    for s in spans:
        if not any(k.type == s.type and s.iou(k) > 0.5 for k in kept):
            kept.append(s)
    return sorted(kept, key=lambda s: s.start)

# ---------------------------------------------------------------------------
# Presidio baseline (sm / lg spaCy NLP engine), mapped to the canonical types.
# DATE_TIME->DOB and LOCATION->ADDRESS are deliberately recall-generous.
# ---------------------------------------------------------------------------
PRESIDIO_TO_CANON = {
    "PERSON": "PERSON", "EMAIL_ADDRESS": "EMAIL", "PHONE_NUMBER": "PHONE",
    "US_SSN": "SSN", "CREDIT_CARD": "CREDIT_CARD", "IBAN_CODE": "IBAN",
    "US_BANK_NUMBER": "ACCOUNT_NUMBER", "DATE_TIME": "DOB", "LOCATION": "ADDRESS",
}

class PresidioEngine:
    def __init__(self, model: str = "en_core_web_sm"):
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        conf = {"nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": model}]}
        self.name = f"presidio-{model.rsplit('_',1)[-1]}"
        self.analyzer = AnalyzerEngine(
            nlp_engine=NlpEngineProvider(nlp_configuration=conf).create_engine())

    def detect(self, text: str) -> list[Span]:
        res = self.analyzer.analyze(text=text, language="en")
        out = []
        for r in res:
            t = PRESIDIO_TO_CANON.get(r.entity_type)
            if t:
                out.append(Span(r.start, r.end, t, text[r.start:r.end], float(r.score)))
        return _dedupe_overlaps(out)

    def detect_raw(self, text: str) -> list[Span]:
        """Unmapped Presidio types, for PIIBench-native evaluation."""
        return [Span(r.start, r.end, r.entity_type, text[r.start:r.end], float(r.score))
                for r in self.analyzer.analyze(text=text, language="en")]

# ---------------------------------------------------------------------------
# LLM engine: strict JSON span contract. Anthropic Messages API (or any
# OpenAI-compatible endpoint via base_url). Requires an API key at runtime;
# the harness step `--engine llm` is the only step that needs network+key.
# ---------------------------------------------------------------------------
LLM_PROMPT = """Extract every personally identifiable information span from the document.
Return ONLY a JSON array, no prose. Each element:
{"text": "<verbatim span>", "type": "<one of ACCOUNT_NUMBER|ADDRESS|CREDIT_CARD|DOB|EMAIL|IBAN|INSURANCE_ID|MRN|PERSON|PHONE|SSN>", "confidence": <0..1>}
Spans must be verbatim substrings of the document. Document:
"""

class LLMEngine:
    name = "llm"
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6",
                 base_url: str = "https://api.anthropic.com/v1/messages"):
        self.key, self.model, self.url = api_key, model, base_url

    def detect(self, text: str) -> list[Span]:
        body = json.dumps({"model": self.model, "max_tokens": 2000,
                           "messages": [{"role": "user", "content": LLM_PROMPT + text}]}).encode()
        req = urllib.request.Request(self.url, data=body, headers={
            "content-type": "application/json", "x-api-key": self.key,
            "anthropic-version": "2023-06-01"})
        raw = json.loads(urllib.request.urlopen(req, timeout=120).read())
        payload = "".join(b.get("text", "") for b in raw.get("content", []))
        payload = re.sub(r"```(?:json)?|```", "", payload).strip()
        out = []
        try:
            for item in json.loads(payload):
                idx = text.find(item.get("text", ""))
                if idx >= 0 and item.get("type") in LLM_PROMPT:
                    out.append(Span(idx, idx + len(item["text"]), item["type"],
                                    item["text"], float(item.get("confidence", 0.5))))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        return _dedupe_overlaps(out)
