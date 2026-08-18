"""Core: taxonomy, split conformal gate, dedup, surrogate anonymizer, metrics.

Implements the framework of "Conformally Gated Agents" (CGA). All numbers in
the revised paper are emitted by run_experiments.py using this module.
"""
from __future__ import annotations
import hashlib, hmac, math, random, re
from collections import defaultdict
from dataclasses import dataclass, field

CANONICAL_TYPES = ["ACCOUNT_NUMBER","ADDRESS","CREDIT_CARD","DOB","EMAIL",
                   "IBAN","INSURANCE_ID","MRN","PERSON","PHONE","SSN"]
POOLED = "__POOLED__"

@dataclass
class Span:
    start: int; end: int; type: str; text: str = ""; score: float = 0.0
    def iou(self, o: "Span") -> float:
        inter = max(0, min(self.end, o.end) - max(self.start, o.start))
        if inter == 0: return 0.0
        union = (self.end - self.start) + (o.end - o.start) - inter
        return inter / union

# ---------------- Split conformal gate (Mondrian per-type + pooled) --------
class ConformalGate:
    """Algorithm 1 of the paper. attach(): score of best-overlapping same-type
    prediction, else 0. threshold tau_y = 1 - qhat_y with
    qhat_y = ceil((n+1)(1-alpha))/n empirical quantile of r_i = 1 - s_i
    (capped at 1.0 when the index exceeds n, giving tau_y = 0: transparent
    degradation)."""
    def __init__(self, alpha: float = 0.10, mondrian: bool = True):
        self.alpha = alpha; self.mondrian = mondrian
        self.tau: dict[str, float] = {}; self.n: dict[str, int] = {}

    @staticmethod
    def attach_scores(gold: list[Span], preds: list[Span],
                      iou_thr: float = 0.5) -> list[tuple[str, float]]:
        """Score of best same-type prediction overlapping at IoU >= iou_thr,
        else 0. The threshold matches the leak/mask definition of Cor. 1."""
        out = []
        for g in gold:
            best = 0.0
            for p in preds:
                if p.type == g.type and g.iou(p) >= iou_thr:
                    best = max(best, p.score)
            out.append((g.type, best))
        return out

    def fit(self, attached: list[tuple[str, float]]):
        buckets: dict[str, list[float]] = defaultdict(list)
        for t, s in attached:
            buckets[t if self.mondrian else POOLED].append(1.0 - s)
        pooled_all = [1.0 - s for _, s in attached]
        self._fit_bucket(POOLED, pooled_all)
        if self.mondrian:
            for t, rs in buckets.items():
                self._fit_bucket(t, rs)
        return self

    def _fit_bucket(self, key: str, rs: list[float]):
        n = len(rs)
        self.n[key] = n
        if n == 0:
            self.tau[key] = 0.0; return
        rs = sorted(rs)
        k = math.ceil((n + 1) * (1 - self.alpha))       # 1-indexed rank
        qhat = 1.0 if k > n else rs[k - 1]
        self.tau[key] = 1.0 - qhat

    def threshold(self, t: str) -> float:
        if self.mondrian and t in self.tau: return self.tau[t]
        return self.tau.get(POOLED, 0.0)

    def decide(self, span: Span, review_floor: float) -> str:
        tau = self.threshold(span.type)
        if span.score >= tau: return "accept"
        if span.score >= review_floor: return "review"
        return "reject"

# ---------------- Deduplication: SHA-256 + MinHash/LSH ---------------------
def canon(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

def shingles(text: str, k: int = 5) -> set[bytes]:
    t = canon(text)
    if len(t) < k: return {t.encode()}
    return {t[i:i+k].encode() for i in range(len(t) - k + 1)}

class Deduplicator:
    """Online exact (SHA-256) + near (MinHash 128 perms, LSH b=32,r=4,
    verify at theta) duplicate suppression over the processed stream."""
    def __init__(self, theta: float = 0.85, num_perm: int = 128, bands=(32, 4)):
        from datasketch import MinHash, MinHashLSH
        self._MinHash = MinHash
        self.theta = theta; self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=theta, num_perm=num_perm, params=bands)
        self.hashes: set[str] = set(); self.sigs: dict[str, object] = {}; self._i = 0

    def _mh(self, text: str):
        m = self._MinHash(num_perm=self.num_perm)
        for s in shingles(text): m.update(s)
        return m

    def check_and_add(self, text: str) -> tuple[bool, str]:
        """Returns (is_duplicate, kind in {'', 'exact', 'near'}). Adds firsts."""
        h = hashlib.sha256(canon(text).encode()).hexdigest()
        if h in self.hashes: return True, "exact"
        m = self._mh(text)
        for key in self.lsh.query(m):                      # candidate pairs
            if self.sigs[key].jaccard(m) >= self.theta:    # verify on signature
                return True, "near"
        self.hashes.add(h)
        key = f"d{self._i}"; self._i += 1
        self.lsh.insert(key, m); self.sigs[key] = m
        return False, ""

# ---------------- Deterministic keyed surrogates ----------------------------
FIRST = ["Alex","Jordan","Riley","Casey","Morgan","Avery","Quinn","Rowan","Sage","Ellis",
         "Harper","Reese","Emerson","Finley","Skyler","Dakota","Peyton","Kendall","Marlow","Tatum"]
LAST  = ["Calder","Whitfield","Norwood","Ashcombe","Fenwick","Larkspur","Mercer","Holloway",
         "Pemberton","Rutledge","Standish","Thornbury","Vexley","Winslow","Yardley","Bexford",
         "Cranmore","Dunmore","Everhart","Farrow"]
STREETS = ["Maple","Cedar","Willow","Aspen","Birch","Juniper","Laurel","Rowanberry","Hazel","Alder"]
SUFFIX  = ["St","Ave","Rd","Ln","Blvd","Dr","Ct","Way"]

def _luhn_check_digit(digits: str) -> str:
    total, dbl = 0, True
    for d in reversed(digits):
        v = int(d)
        if dbl:
            v *= 2
            if v > 9: v -= 9
        total += v; dbl = not dbl
    return str((10 - total % 10) % 10)

def luhn_valid(digits: str) -> bool:
    digits = re.sub(r"\D", "", digits)
    if len(digits) < 12: return False
    return _luhn_check_digit(digits[:-1]) == digits[-1]

def iban_valid(s: str) -> bool:
    s = re.sub(r"\s", "", s).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", s): return False
    r = s[4:] + s[:4]
    num = "".join(str(int(c, 36)) for c in r)
    return int(num) % 97 == 1

class Anonymizer:
    """HMAC-keyed, deterministic, format- and checksum-consistent surrogates."""
    def __init__(self, key: bytes = b"cga-release-key"):
        self.key = key

    def _rng(self, text: str) -> random.Random:
        d = hmac.new(self.key, text.encode(), hashlib.sha256).digest()
        return random.Random(int.from_bytes(d[:8], "big"))

    def surrogate(self, span: Span) -> str:
        rng, t, s = self._rng(span.text), span.type, span.text
        if t == "EMAIL":
            dom = s.split("@")[-1] if "@" in s else "example.org"
            return f"{rng.choice(FIRST).lower()}.{rng.choice(LAST).lower()}@{dom}"
        if t == "PERSON":
            return f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        if t == "PHONE":
            return re.sub(r"\d", lambda _: str(rng.randint(0, 9)), s)
        if t == "SSN":
            return f"{rng.randint(100,665):03d}-{rng.randint(1,99):02d}-{rng.randint(1,9999):04d}"
        if t == "CREDIT_CARD":
            body = "".join(str(rng.randint(0, 9)) for _ in range(15))
            digits = body + _luhn_check_digit(body)
            out, di = [], 0
            for ch in s:
                if ch.isdigit():
                    out.append(digits[di % 16]); di += 1
                else: out.append(ch)
            return "".join(out)
        if t == "IBAN":
            cc = s[:2].upper() if s[:2].isalpha() else "DE"
            body = "".join(str(rng.randint(0, 9)) for _ in range(16))
            for chk in range(2, 99):
                cand = f"{cc}{chk:02d}{body}"
                if iban_valid(cand): return cand
            return f"{cc}00{body}"
        if t == "DOB":
            return f"{rng.randint(1,28):02d}/{rng.randint(1,12):02d}/{rng.randint(1940,2005)}"
        if t == "ADDRESS":
            return f"{rng.randint(10,9999)} {rng.choice(STREETS)} {rng.choice(SUFFIX)}"
        if t in ("ACCOUNT_NUMBER","MRN","INSURANCE_ID"):
            return re.sub(r"[0-9]", lambda _: str(rng.randint(0, 9)),
                   re.sub(r"[A-Z]", lambda _: chr(rng.randint(65, 90)), s))
        return "".join(rng.choice("XZQJ0123456789") for _ in s)

    def apply(self, text: str, spans: list[Span]) -> str:
        for sp in sorted(spans, key=lambda x: -x.start):   # right to left
            text = text[:sp.start] + self.surrogate(sp) + text[sp.end:]
        return text

# ---------------- Metrics ---------------------------------------------------
def match_spans(gold: list[Span], pred: list[Span], iou_thr: float = 0.5):
    """Greedy 1-1 matching: same type, IoU >= thr, highest IoU first."""
    pairs = sorted(((g.iou(p), gi, pi) for gi, g in enumerate(gold)
                    for pi, p in enumerate(pred)
                    if p.type == g.type and g.iou(p) >= iou_thr),
                   key=lambda x: -x[0])
    ug, up, matched = set(), set(), []
    for _, gi, pi in pairs:
        if gi in ug or pi in up: continue
        ug.add(gi); up.add(pi); matched.append((gi, pi))
    return matched, ug, up

def prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f

def evaluate(docs: list[dict], iou_thr: float = 0.5):
    """docs: [{gold:[Span], pred:[Span]}]. Returns micro + per-type PRF."""
    tp = fp = fn = 0
    per = defaultdict(lambda: [0, 0, 0])       # type -> [tp, fp, fn]
    for d in docs:
        m, ug, up = match_spans(d["gold"], d["pred"], iou_thr)
        tp += len(m); fn += len(d["gold"]) - len(ug); fp += len(d["pred"]) - len(up)
        for gi, _ in m: per[d["gold"][gi].type][0] += 1
        for gi, g in enumerate(d["gold"]):
            if gi not in ug: per[g.type][2] += 1
        for pi, p in enumerate(d["pred"]):
            if pi not in up: per[p.type][1] += 1
    micro = dict(zip(("P","R","F1"), prf(tp, fp, fn)))
    micro.update(tp=tp, fp=fp, fn=fn)
    per_type = {t: dict(zip(("P","R","F1"), prf(*v)), support=v[0] + v[2])
                for t, v in sorted(per.items())}
    return {"micro": micro, "per_type": per_type}

def bootstrap_ci(docs: list[dict], stat: str = "F1", reps: int = 1000,
                 seed: int = 0, iou_thr: float = 0.5):
    rng = random.Random(seed); n = len(docs); vals = []
    for _ in range(reps):
        sample = [docs[rng.randrange(n)] for _ in range(n)]
        vals.append(evaluate(sample, iou_thr)["micro"][stat])
    vals.sort()
    return vals[int(0.025 * reps)], vals[int(0.975 * reps)]

def utility_retention(orig: str, anon: str, gold: list[Span]) -> float:
    """Token-level Jaccard retention of non-PII content."""
    cut, prev = [], 0
    for sp in sorted(gold, key=lambda x: x.start):
        cut.append(orig[prev:sp.start]); prev = sp.end
    cut.append(orig[prev:])
    keep = set(" ".join(cut).split())
    got = set(anon.split())
    if not keep: return 1.0
    return len(keep & got) / len(keep)
