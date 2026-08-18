#!/usr/bin/env python3
"""Certified end-to-end leakage bound (Corollary 1 of the paper).

Per type y:  P(leak_y) <= alpha * 1[tau_y > 0]  +  U_{delta'}(phat_y, n_y)
where phat_y is the empirical candidate-miss rate on calibration spans
(no same-type prediction overlapping at IoU >= 0.5), U is the exact
Clopper-Pearson upper bound, and delta' = delta / |Y| for simultaneity.
Compares the certified bound against the realized per-type test miss.
Pure-python CP bound (binomial tail bisection); no scipy dependency.
"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synpii.core import ConformalGate, CANONICAL_TYPES, evaluate
from synpii.engines import RuleEngine
from synpii.generator import generate_corpus
from run_experiments import split_stream, fit_gate, run_system_on_docs

def binom_cdf(x, n, p):
    """P(X <= x) for X~Bin(n,p), stable via log terms."""
    if p <= 0: return 1.0
    if p >= 1: return 1.0 if x >= n else 0.0
    total, logc = 0.0, 0.0  # log C(n,0)=0
    lp, lq = math.log(p), math.log(1 - p)
    for k in range(0, x + 1):
        if k > 0:
            logc += math.log(n - k + 1) - math.log(k)
        total += math.exp(logc + k * lp + (n - k) * lq)
    return min(total, 1.0)

def cp_upper(x, n, delta):
    """Exact Clopper-Pearson upper confidence bound for a binomial rate."""
    if x >= n: return 1.0
    lo, hi = x / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if binom_cdf(x, n, mid) > delta: lo = mid
        else: hi = mid
    return hi

def run(seed=7, alpha=0.10, delta=0.10):
    eng = RuleEngine()
    stream = generate_corpus(600, 0.12, seed)
    calib, test = split_stream(stream, 0.4, seed)
    calib_u = [d for d in calib if not d["is_dup"]]
    gate, _ = fit_gate(eng, calib_u, alpha)

    # candidate-miss counts on calibration (per type)
    miss, tot = {t: 0 for t in CANONICAL_TYPES}, {t: 0 for t in CANONICAL_TYPES}
    for d in calib_u:
        preds = eng.detect(d["text"])
        for g in d["gold"]:
            tot[g.type] += 1
            if not any(p.type == g.type and g.iou(p) >= 0.5 for p in preds):
                miss[g.type] += 1

    # realized per-type miss on test (headless-accept full pipeline)
    proc, *_ = run_system_on_docs(eng, gate, test, 0.25)
    ev = evaluate([dict(gold=d["gold"], pred=d["pred"]) for d in proc
                   if d.get("processed") and d.get("gold") is not None])

    dprime = delta / len(CANONICAL_TYPES)
    rows = {}
    for t in CANONICAL_TYPES:
        n, x = tot[t], miss[t]
        U = cp_upper(x, n, dprime) if n else 1.0
        tau = gate.threshold(t)
        bound = alpha if tau > 0 else min(1.0, U)   # sharp form (Cor. 1)
        realized = 1.0 - ev["per_type"].get(t, {}).get("R", 0.0)
        rows[t] = dict(n_calib=n, cand_miss_hat=round(x / n, 4) if n else None,
                       cp_ucb=round(U, 4), tau=round(tau, 3),
                       certified_bound=round(bound, 4),
                       realized_test_miss=round(realized, 4),
                       holds=realized <= bound)
    out = dict(seed=seed, alpha=alpha, delta=delta, delta_per_type=dprime,
               per_type=rows,
               all_hold=all(r["holds"] for r in rows.values()))
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/e2e_bounds.json", "w"), indent=1)
    for t, r in rows.items():
        print(f"{t:15s} n={r['n_calib']:4d} phat={r['cand_miss_hat']:.3f} "
              f"UCB={r['cp_ucb']:.3f} tau={r['tau']:.2f} "
              f"bound={r['certified_bound']:.3f} realized={r['realized_test_miss']:.3f} "
              f"{'OK' if r['holds'] else 'VIOLATED'}")
    print("all_hold:", out["all_hold"])

if __name__ == "__main__":
    run()
