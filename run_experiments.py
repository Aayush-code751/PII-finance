#!/usr/bin/env python3
"""CGA experiment harness. Steps:
  synthetic   multi-seed SynPII-FH eval: ours(rule)+gate vs Presidio sm/lg
  ablations   conformal vs fixed thresholds, alpha sweep, Mondrian vs pooled,
              calibration-size sweep, review-floor sweep
  dedupstress same-template different-entity false-suppression test
  piibench    official-protocol evaluation (needs test/calibration jsonl)
  llm         LLM engine on any corpus (needs ANTHROPIC_API_KEY)
All results land in results/*.json; every number in the paper cites a key here.
"""
from __future__ import annotations
import argparse, json, os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synpii.core import (Span, ConformalGate, Deduplicator, Anonymizer,
                         evaluate, bootstrap_ci, utility_retention, CANONICAL_TYPES)
from synpii.engines import RuleEngine, PresidioEngine, LLMEngine
from synpii.generator import generate_corpus, stress_pairs
from synpii import piibench as pb
from synpii import external_datasets as ext
from synpii.graph import build_pipeline

R = "results"; os.makedirs(R, exist_ok=True)
def save(name, obj):
    with open(f"{R}/{name}.json", "w") as f: json.dump(obj, f, indent=1, default=str)
    print(f"  -> {R}/{name}.json")

def split_stream(stream, calib_frac=0.4, seed=7):
    import random
    uniq = [d for d in stream if not d["is_dup"]]
    rng = random.Random(seed); rng.shuffle(uniq)
    k = int(len(uniq) * calib_frac)
    calib_ids = {d["id"] for d in uniq[:k]}
    calib = [d for d in stream if d["id"] in calib_ids]
    test  = [d for d in stream if d["id"] not in calib_ids]   # dups ride with test
    return calib, test

def run_system_on_docs(engine, gate, docs, review_floor, review_policy="accept",
                       with_dedup=True, anonymize=True):
    dedup = Deduplicator() if with_dedup else None
    anon = Anonymizer()
    out, n_sup, sup_kinds = [], 0, {"exact": 0, "near": 0}
    t0 = time.perf_counter()
    for d in docs:
        if dedup:
            is_dup, kind = dedup.check_and_add(d["text"])
            if is_dup:
                n_sup += 1; sup_kinds[kind] += 1
                out.append(dict(d, processed=False)); continue
        preds = engine.detect(d["text"])
        accepted, review = [], []
        if gate:
            for sp in preds:
                dec = gate.decide(sp, review_floor)
                if dec == "accept": accepted.append(sp)
                elif dec == "review": review.append(sp)
            if review_policy == "accept": accepted += review
        else:
            accepted = preds
        rec = dict(d, processed=True, pred=accepted, n_review=len(review),
                   raw_pred=preds)
        if anonymize and d.get("gold") is not None:
            rec["anon"] = anon.apply(d["text"], accepted)
        out.append(rec)
    dt = time.perf_counter() - t0
    return out, n_sup, sup_kinds, dt

def summarize(processed, gate, dt):
    ev_docs = [dict(gold=d["gold"], pred=d["pred"]) for d in processed
               if d.get("processed") and d.get("gold") is not None]
    res = evaluate(ev_docs)
    # end-to-end miss rate (fraction of gold spans NOT masked) — the honest
    # leakage number, distinct from conformal coverage
    total_gold = sum(len(d["gold"]) for d in ev_docs)
    missed = res["micro"]["fn"]
    res["end_to_end_miss_rate"] = missed / total_gold if total_gold else 0.0
    if gate:
        attached = []
        for d in ev_docs:
            attached += ConformalGate.attach_scores(
                d["gold"], [s for dd in [d] for s in dd["pred"]])
        # coverage against calibrated thresholds uses raw engine scores:
        cov_n = cov_ok = 0
        for d in processed:
            if not d.get("processed") or d.get("gold") is None: continue
            att = ConformalGate.attach_scores(d["gold"], d["raw_pred"])
            for t, s in att:
                cov_n += 1
                if s >= gate.threshold(t): cov_ok += 1
        res["coverage"] = cov_ok / cov_n if cov_n else None
        res["thresholds"] = {t: round(gate.threshold(t), 3) for t in CANONICAL_TYPES}
        res["calib_n"] = dict(gate.n)
    ut = [utility_retention(d["text"], d["anon"], d["gold"])
          for d in processed if d.get("anon")]
    res["utility"] = sum(ut) / len(ut) if ut else None
    res["docs_per_s"] = len(processed) / dt if dt else None
    res["review_load_per_doc"] = (statistics.mean(d.get("n_review", 0)
        for d in processed if d.get("processed")) if processed else 0)
    return res

def fit_gate(engine, calib_docs, alpha, mondrian=True):
    attached = []
    for d in calib_docs:
        preds = engine.detect(d["text"])
        attached += ConformalGate.attach_scores(d["gold"], preds)
    return ConformalGate(alpha=alpha, mondrian=mondrian).fit(attached), len(attached)

# ------------------------------- steps --------------------------------------
def step_synthetic(seeds=(7, 13, 21, 42, 77), alpha=0.10, floor=0.25):
    print("== synthetic multi-seed ==")
    systems_out = {}
    engines = {"ours-rule": RuleEngine(),
               "presidio-sm": PresidioEngine("en_core_web_sm")}
    try:
        engines["presidio-lg"] = PresidioEngine("en_core_web_lg")
    except Exception as e:
        print("  presidio-lg unavailable:", e)
    per_seed = {k: [] for k in engines}
    detail = {}
    for seed in seeds:
        stream = generate_corpus(600, 0.12, seed)
        calib, test = split_stream(stream, 0.4, seed)
        calib_u = [d for d in calib if not d["is_dup"]]
        for name, eng in engines.items():
            gate = None
            if name == "ours-rule":
                gate, n_att = fit_gate(eng, calib_u, alpha)
            proc, n_sup, kinds, dt = run_system_on_docs(
                eng, gate, test, floor, with_dedup=(name == "ours-rule"))
            res = summarize(proc, gate, dt)
            res.update(seed=seed, suppressed=n_sup, sup_kinds=kinds,
                       n_test_docs=len(test))
            per_seed[name].append(res)
            if seed == seeds[0]:
                res["F1_ci95"] = bootstrap_ci(
                    [dict(gold=d["gold"], pred=d["pred"]) for d in proc
                     if d.get("processed") and d.get("gold") is not None])
                detail[name] = res
            print(f"  seed {seed} {name}: F1={res['micro']['F1']:.3f} "
                  f"P={res['micro']['P']:.3f} R={res['micro']['R']:.3f} "
                  f"miss={res['end_to_end_miss_rate']:.3f}")
    agg = {}
    for name, runs in per_seed.items():
        agg[name] = {m: dict(
            mean=statistics.mean(r["micro"][m] for r in runs),
            sd=statistics.pstdev(r["micro"][m] for r in runs))
            for m in ("P", "R", "F1")}
        agg[name]["end_to_end_miss_rate"] = dict(
            mean=statistics.mean(r["end_to_end_miss_rate"] for r in runs),
            sd=statistics.pstdev(r["end_to_end_miss_rate"] for r in runs))
    save("synthetic_multiseed", dict(aggregate=agg, per_seed=per_seed,
                                     seed7_detail=detail))

def step_ablations(seed=7, floor=0.25):
    print("== ablations (seed 7) ==")
    eng = RuleEngine()
    stream = generate_corpus(600, 0.12, seed)
    calib, test = split_stream(stream, 0.4, seed)
    calib_u = [d for d in calib if not d["is_dup"]]
    out = {}
    # A: conformal (alpha sweep) vs fixed thresholds vs no gate
    for alpha in (0.05, 0.10, 0.20):
        gate, _ = fit_gate(eng, calib_u, alpha)
        proc, *_ , dt = run_system_on_docs(eng, gate, test, floor)
        out[f"conformal_a{alpha}"] = summarize(proc, gate, dt)
    for thr in (0.5, 0.85):
        class Fixed:  # fixed-threshold pseudo-gate
            def __init__(s, t): s.t = t; s.n = {}
            def threshold(s, _): return s.t
            def decide(s, sp, floor):
                return "accept" if sp.score >= s.t else ("review" if sp.score >= floor else "reject")
        g = Fixed(thr)
        proc, *_ , dt = run_system_on_docs(eng, g, test, floor)
        out[f"fixed_{thr}"] = summarize(proc, g, dt)
    proc, *_ , dt = run_system_on_docs(eng, None, test, floor)
    out["no_gate"] = summarize(proc, None, dt)
    # B: Mondrian vs pooled
    for mond in (True, False):
        gate, _ = fit_gate(eng, calib_u, 0.10, mondrian=mond)
        proc, *_ , dt = run_system_on_docs(eng, gate, test, floor)
        out[f"{'mondrian' if mond else 'pooled'}_a0.10"] = summarize(proc, gate, dt)
    # C: calibration-size sensitivity
    import random as _r
    for frac in (0.25, 0.5, 1.0):
        sub = _r.Random(1).sample(calib_u, max(5, int(len(calib_u) * frac)))
        gate, _ = fit_gate(eng, sub, 0.10)
        proc, *_ , dt = run_system_on_docs(eng, gate, test, floor)
        s = summarize(proc, gate, dt); s["n_calib_docs"] = len(sub)
        out[f"calibfrac_{frac}"] = s
    # D: review floor sweep (headless-accept and headless-reject bounds)
    for b in (0.10, 0.25, 0.40):
        gate, _ = fit_gate(eng, calib_u, 0.10)
        for pol in ("accept", "reject"):
            proc, *_ , dt = run_system_on_docs(eng, gate, test, b, review_policy=pol)
            out[f"floor_{b}_{pol}"] = summarize(proc, gate, dt)
    save("ablations_seed7", out)

def _pair_jaccard(a, b):
    from synpii.core import shingles
    A, B = shingles(a), shingles(b)
    return len(A & B) / len(A | B)

def step_dedupstress(seed=7):
    print("== dedup stress ==")
    res = {}
    for cond in ("resampled_filler", "minimal_pair"):
        pairs = stress_pairs(150, seed, minimal=(cond == "minimal_pair"))
        jacc = sorted(_pair_jaccard(a, b) for a, b in pairs)
        res[f"jaccard_{cond}"] = dict(
            min=jacc[0], p50=jacc[len(jacc)//2], p95=jacc[int(0.95*len(jacc))],
            max=jacc[-1])
        for theta in (0.70, 0.80, 0.85, 0.90, 0.95):
            fs = 0
            for a, b in pairs:
                dd = Deduplicator(theta=theta)
                dd.check_and_add(a)
                dup, _ = dd.check_and_add(b)
                fs += dup
            res[f"{cond}_theta{theta}"] = dict(
                false_suppression_rate=fs / len(pairs), n_pairs=len(pairs))
            print(f"  {cond} theta={theta}: false-suppression={fs/len(pairs):.3f}")
    # order-aware detection of injected duplicates: a duplicate only counts
    # toward the denominator if its source appeared EARLIER in the stream
    # (the online index correctly admits stream-first copies).
    for theta in (0.80, 0.85, 0.90):
        stream = generate_corpus(200, 0.12, seed)
        dd = Deduplicator(theta=theta)
        seen, seen_dup_srcs = set(), set()
        det = {"near": [0, 0], "exact": [0, 0]}; false_sup = 0
        for d in stream:
            dup, kind = dd.check_and_add(d["text"])
            if d["is_dup"] and d["src_id"] in seen:
                det[d["dup_kind"]][1] += 1; det[d["dup_kind"]][0] += dup
            elif not d["is_dup"] and dup and d["id"] not in seen_dup_srcs:
                false_sup += 1     # suppressed with no earlier copy: FALSE
            if d["is_dup"]: seen_dup_srcs.add(d["src_id"])
            seen.add(d["id"])
        res[f"stream_theta{theta}"] = dict(
            near_detect=det["near"][0] / det["near"][1] if det["near"][1] else None,
            exact_detect=det["exact"][0] / det["exact"][1] if det["exact"][1] else None,
            n_near_in_order=det["near"][1], n_exact_in_order=det["exact"][1],
            false_suppressions_on_uniques=false_sup)
        print(f"  stream theta={theta}: near={res[f'stream_theta{theta}']['near_detect']} "
              f"exact={res[f'stream_theta{theta}']['exact_detect']} "
              f"false-sup={false_sup}")
    save("dedup_stress", res)

def step_graph_smoke(seed=7):
    """End-to-end LangGraph run to certify the pipeline (audit trail incl.)."""
    print("== langgraph pipeline smoke ==")
    eng = RuleEngine()
    stream = generate_corpus(60, 0.12, seed)
    calib, test = split_stream(stream, 0.4, seed)
    gate, _ = fit_gate(eng, [d for d in calib if not d["is_dup"]], 0.10)
    pipe = build_pipeline(eng, gate, Deduplicator(), Anonymizer())
    audits, n_sup = [], 0
    for d in test[:40]:
        st = pipe.invoke({"doc_id": d["id"], "text": d["text"]})
        n_sup += bool(st.get("suppressed"))
        audits.append([a["node"] for a in st.get("audit", [])])
    save("graph_smoke", dict(n_docs=40, n_suppressed=n_sup,
                             sample_audit_paths=audits[:5]))

def step_piibench(test_file, calib_file, subset_size=5000, seed=42,
                  alpha=0.10, floor=0.25, presidio_model="en_core_web_lg"):
    print("== PIIBench official-protocol evaluation ==")
    test_recs = pb.load_jsonl(test_file)
    subset = pb.make_stratified_subset(test_recs, subset_size, seed)
    prep, bad = pb.prepare_records(subset)
    calib_prep, cbad = pb.prepare_records(
        pb.make_stratified_subset(pb.load_jsonl(calib_file), max(2000, subset_size // 2), seed))
    print(f"  test={len(prep)} (align-fail {bad}), calib={len(calib_prep)} (align-fail {cbad})")
    rule = RuleEngine(); pres = PresidioEngine(presidio_model)
    # calibrate on canonical gold from the calibration split
    attached = []
    for d in calib_prep:
        attached += ConformalGate.attach_scores(d["gold_canon"], rule.detect(d["text"]))
    gate = ConformalGate(alpha=alpha).fit(attached)
    out = {"protocol": dict(subset_size=len(prep), seed=seed,
                            sampler="source-stratified largest-remainder (upstream)",
                            metric_a="char-span IoU>=0.5 canonical taxonomy",
                            metric_b="seqeval exact span+type on PIIBench labels"),
           "thresholds": {t: round(gate.threshold(t), 3) for t in CANONICAL_TYPES},
           "calib_n_per_type": dict(gate.n)}
    for name, eng, use_gate in (("ours-rule+gate", rule, True),
                                 ("presidio", pres, False)):
        docs, t0 = [], time.perf_counter()
        cov_n = cov_ok = 0
        for d in prep:
            preds = eng.detect(d["text"])
            if use_gate:
                kept = [p for p in preds if gate.decide(p, floor) != "reject"]
                for t, s in ConformalGate.attach_scores(d["gold_canon"], preds):
                    cov_n += 1; cov_ok += (s >= gate.threshold(t))
            else:
                kept = preds
            docs.append(dict(gold=d["gold_canon"], pred=kept))
        dt = time.perf_counter() - t0
        res = evaluate(docs)
        res["docs_per_s"] = len(prep) / dt
        res["F1_ci95"] = bootstrap_ci(docs)
        if use_gate:
            res["coverage"] = cov_ok / cov_n if cov_n else None
            total_gold = sum(len(d["gold"]) for d in docs)
            res["end_to_end_miss_rate"] = res["micro"]["fn"] / total_gold
        out[name] = res
        print(f"  {name}: F1={res['micro']['F1']:.3f} P={res['micro']['P']:.3f} "
              f"R={res['micro']['R']:.3f}")
    # candidate-recall table on ALL PIIBench gold types (transparency table)
    per_gold_type = {}
    for d in prep:
        preds = rule.detect(d["text"])
        for g in d["gold_piib"]:
            row = per_gold_type.setdefault(g.type, [0, 0])
            row[1] += 1
            canon_t = pb.PIIB_TO_CANON.get(g.type)
            if canon_t and any(p.type == canon_t and g.iou(p) >= 0.5 for p in preds):
                row[0] += 1
    out["candidate_recall_by_piibench_type"] = {
        t: dict(recall=c / n, n=n) for t, (c, n) in
        sorted(per_gold_type.items(), key=lambda kv: -kv[1][1])}
    save("piibench_official", out)

def _cap_length(docs, max_chars=8000):
    """Truncate documents (and drop gold spans past the cutoff) so a handful
    of pathologically long outliers cannot dominate wall-clock time. Applied
    uniformly to every engine so the comparison stays fair; the truncated
    count is reported."""
    out, n_trunc = [], 0
    for d in docs:
        text = d["text"]
        if len(text) <= max_chars:
            out.append(d); continue
        n_trunc += 1
        gold = [s for s in d["gold"] if s.end <= max_chars]
        out.append(dict(d, text=text[:max_chars], gold=gold))
    return out, n_trunc

def step_external(name, docs, alpha=0.10, floor=0.25, calib_frac=0.4, seed=42,
                  presidio_model="en_core_web_lg", synth_seed=7, max_chars=8000):
    """Third-party validation on a real external corpus (docs already in
    {id,text,gold:[Span]} shape via synpii/external_datasets.py). Reports two
    conditions, neither invented: (a) recalibrated -- gate fit fresh on a
    held-out slice of this same corpus; (b) transfer -- the gate calibrated on
    the synthetic corpus (seed 7) applied as-is, measuring what the paper's
    Limitations section only asserted in prose about distribution shift."""
    docs, n_trunc = _cap_length(docs, max_chars)
    print(f"== external: {name} ({len(docs)} docs, {n_trunc} truncated to "
          f"{max_chars} chars) ==")
    import random as _random
    rng = _random.Random(seed)
    shuffled = list(docs); rng.shuffle(shuffled)
    k = int(len(shuffled) * calib_frac)
    calib, test = shuffled[:k], shuffled[k:]

    rule = RuleEngine()
    try:
        pres = PresidioEngine(presidio_model)
    except Exception as e:
        pres = None
        print("  presidio unavailable:", e)

    attached = []
    for d in calib:
        attached += ConformalGate.attach_scores(d["gold"], rule.detect(d["text"]))
    gate_recal = ConformalGate(alpha=alpha).fit(attached)

    synth_stream = generate_corpus(600, 0.12, synth_seed)
    synth_calib, _ = split_stream(synth_stream, 0.4, synth_seed)
    synth_calib_u = [d for d in synth_calib if not d["is_dup"]]
    gate_transfer, _ = fit_gate(rule, synth_calib_u, alpha)

    # Cache raw predictions on the test set ONCE per engine -- neither the
    # rule engine's nor Presidio's predictions depend on which gate
    # (recalibrated vs. transfer) is applied afterward, so re-running
    # detect() per condition was pure duplicated work.
    print(f"  detecting with rule engine ({len(test)} docs)...")
    rule_preds = [rule.detect(d["text"]) for d in test]
    pres_preds = None
    if pres is not None:
        print(f"  detecting with presidio ({len(test)} docs)...")
        pres_preds = [pres.detect(d["text"]) for d in test]

    presidio_result = None
    if pres_preds is not None:
        t0 = time.perf_counter()
        docs_eval = [dict(gold=d["gold"], pred=p) for d, p in zip(test, pres_preds)]
        dt = time.perf_counter() - t0
        presidio_result = evaluate(docs_eval)
        presidio_result["docs_per_s"] = len(test) / dt if dt else None
        presidio_result["F1_ci95"] = bootstrap_ci(docs_eval)
        print(f"  presidio: F1={presidio_result['micro']['F1']:.3f} "
              f"P={presidio_result['micro']['P']:.3f} "
              f"R={presidio_result['micro']['R']:.3f}")

    out = {"protocol": dict(name=name, n_total=len(docs), n_calib=len(calib),
                            n_test=len(test), seed=seed, alpha=alpha,
                            n_truncated=n_trunc, max_chars=max_chars,
                            note="recalibrated = gate fit on this corpus's own "
                                 "calibration slice; transfer = gate calibrated "
                                 "on the synthetic corpus (seed 7), applied as-is; "
                                 "presidio is identical in both since it never "
                                 "consults the gate")}
    for cond_name, gate in (("recalibrated", gate_recal), ("transfer", gate_transfer)):
        cond = {"thresholds": {t: round(gate.threshold(t), 3) for t in CANONICAL_TYPES},
                "calib_n": dict(gate.n)}
        t0 = time.perf_counter()
        docs_eval = []
        cov_n = cov_ok = 0
        for d, preds in zip(test, rule_preds):
            kept = [p for p in preds if gate.decide(p, floor) != "reject"]
            for t, s in ConformalGate.attach_scores(d["gold"], preds):
                cov_n += 1; cov_ok += (s >= gate.threshold(t))
            docs_eval.append(dict(gold=d["gold"], pred=kept))
        dt = time.perf_counter() - t0
        res = evaluate(docs_eval)
        res["docs_per_s"] = len(test) / dt if dt else None
        res["F1_ci95"] = bootstrap_ci(docs_eval)
        total_gold = sum(len(d["gold"]) for d in docs_eval)
        res["coverage"] = cov_ok / cov_n if cov_n else None
        res["end_to_end_miss_rate"] = res["micro"]["fn"] / total_gold if total_gold else None
        cond["ours-rule+gate"] = res
        print(f"  [{cond_name}] ours-rule+gate: F1={res['micro']['F1']:.3f} "
              f"P={res['micro']['P']:.3f} R={res['micro']['R']:.3f}")
        if presidio_result is not None:
            cond["presidio"] = presidio_result
        out[cond_name] = cond
    save(f"external_{name}", out)
    return out

def step_external_all():
    print("== external validation: finance datasets ==")
    recs, stats = ext.load_gretel_finance(target_n=5000)
    print("  gretel_finance stats:", json.dumps(stats))
    step_external("gretel_finance", recs)

    recs, stats = ext.load_nemotron("Finance", target_n=5000)
    print("  nemotron_finance stats:", json.dumps(stats))
    step_external("nemotron_finance", recs)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+",
                    help="synthetic|ablations|dedupstress|graph|piibench")
    ap.add_argument("--test-file"); ap.add_argument("--calib-file")
    ap.add_argument("--subset-size", type=int, default=5000)
    args = ap.parse_args()
    for s in args.steps:
        if s == "synthetic": step_synthetic()
        elif s == "ablations": step_ablations()
        elif s == "dedupstress": step_dedupstress()
        elif s == "graph": step_graph_smoke()
        elif s == "piibench": step_piibench(args.test_file, args.calib_file,
                                            args.subset_size)
        elif s == "external": step_external_all()
