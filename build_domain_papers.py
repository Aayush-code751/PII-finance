#!/usr/bin/env python3
"""Build cga-finance/ and cga-health/ repos and run their experiments.
Everything logged; every paper number traceable to results/*.json."""
import json, os, shutil, statistics, subprocess, sys, time, re

BASE = "/home/claude/cga"
sys.path.insert(0, BASE)
from synpii.core import ConformalGate, Deduplicator, Anonymizer, evaluate, bootstrap_ci, Span
from synpii.engines import RuleEngine, PresidioEngine
from synpii import generator as G
from run_experiments import split_stream, fit_gate, run_system_on_docs, summarize
from e2e_bound import cp_upper

def setup_repo(name, domain):
    root = f"/home/claude/{name}"
    if os.path.exists(root): shutil.rmtree(root)
    for d in ("results", "data", "logs", "paper", "scripts"): os.makedirs(f"{root}/{d}")
    shutil.copytree(f"{BASE}/synpii", f"{root}/synpii")
    for f in ("run_experiments.py", "e2e_bound.py", "llm_engine_eval.html"):
        shutil.copy(f"{BASE}/{f}", root)
    return root

def domain_templates(domain):
    return [t for t in G.TEMPLATES if t[0] == domain]

def gen_domain_corpus(domain, n_unique, dup_frac, seed):
    saved = G.TEMPLATES
    G.TEMPLATES = domain_templates(domain)
    try:
        return G.generate_corpus(n_unique, dup_frac, seed)
    finally:
        G.TEMPLATES = saved

def export_data(root, domain, seeds):
    for seed in seeds:
        with open(f"{root}/data/synpii_{domain}_seed{seed}.jsonl", "w") as f:
            for d in gen_domain_corpus(domain, 300, 0.12, seed):
                rec = dict(id=d["id"], text=d["text"], domain=d["domain"],
                           is_dup=d["is_dup"], dup_kind=d["dup_kind"], src_id=d["src_id"],
                           gold=[dict(start=g.start, end=g.end, type=g.type, text=g.text)
                                 for g in d["gold"]] if d["gold"] is not None else None)
                f.write(json.dumps(rec) + "\n")

# ---- healthcare-only: collapse-triggered relaxed escalation -----------------
RELAXED = {
  "INSURANCE_ID": r"(?:[Ii]nsurance|[Pp]olicy|[Mm]ember|[Pp]lan)\s*(?:ID|[Nn]o\.?|[Nn]umber)?\s*[:#]?\s*([A-Z]{2,4}-?\d{5,10})",
  "MRN": r"(?:MRN|[Mm]edical\s+[Rr]ecord)[^A-Za-z0-9]{0,4}([A-Z]?\d{5,8})",
}
def relaxed_candidates(text, collapsed_types):
    out = []
    for t in collapsed_types:
        pat = RELAXED.get(t)
        if not pat: continue
        for m in re.finditer(pat, text):
            out.append(Span(m.start(1), m.end(1), t, m.group(1), 0.30))
    return out

def run_health_escalation(seed, alpha=0.10, floor=0.25):
    eng = RuleEngine()
    stream = gen_domain_corpus("healthcare", 300, 0.12, seed)
    calib, test = split_stream(stream, 0.4, seed)
    calib_u = [d for d in calib if not d["is_dup"]]
    gate, _ = fit_gate(eng, calib_u, alpha)
    collapsed = [t for t in gate.tau if t != "__POOLED__" and gate.tau[t] == 0.0]
    rows = {}
    for policy in ("accept", "reject"):
        dedup = Deduplicator(); docs = []; esc_total = 0
        for d in test:
            dup, _ = dedup.check_and_add(d["text"])
            if dup or d.get("gold") is None: continue
            preds = eng.detect(d["text"])
            accepted = [p for p in preds if gate.decide(p, floor) == "accept"]
            band = [p for p in preds if gate.decide(p, floor) == "review"]
            # collapse-triggered escalation: relaxed candidates -> review only
            esc = [e for e in relaxed_candidates(d["text"], collapsed)
                   if not any(e.iou(p) >= 0.5 and p.type == e.type for p in preds)]
            esc_total += len(esc) + len(band)
            if policy == "accept": accepted = accepted + band + esc
            docs.append(dict(gold=d["gold"], pred=accepted))
        ev = evaluate(docs)
        total_gold = sum(len(d["gold"]) for d in docs)
        rows[policy] = dict(micro=ev["micro"], per_type=ev["per_type"],
                            end_to_end_miss=ev["micro"]["fn"] / total_gold,
                            review_load_per_doc=esc_total / len(docs))
    # no-escalation reference (gate only, accept policy)
    dedup = Deduplicator(); docs = []
    for d in test:
        dup, _ = dedup.check_and_add(d["text"])
        if dup or d.get("gold") is None: continue
        preds = eng.detect(d["text"])
        kept = [p for p in preds if gate.decide(p, floor) != "reject"]
        docs.append(dict(gold=d["gold"], pred=kept))
    ev0 = evaluate(docs); tg = sum(len(d["gold"]) for d in docs)
    return dict(seed=seed, collapsed_types=collapsed,
                thresholds={k: round(v, 3) for k, v in gate.tau.items()},
                no_escalation=dict(micro=ev0["micro"], per_type=ev0["per_type"],
                                   end_to_end_miss=ev0["micro"]["fn"] / tg),
                escalation=rows)

# ---- shared per-domain battery ----------------------------------------------
def bounds_table(domain, seed, alpha=0.10, delta=0.10):
    eng = RuleEngine()
    stream = gen_domain_corpus(domain, 300, 0.12, seed)
    calib, test = split_stream(stream, 0.4, seed)
    calib_u = [d for d in calib if not d["is_dup"]]
    gate, _ = fit_gate(eng, calib_u, alpha)
    types = sorted({g.type for d in calib_u for g in d["gold"]})
    miss = {t: 0 for t in types}; tot = {t: 0 for t in types}
    for d in calib_u:
        preds = eng.detect(d["text"])
        for g in d["gold"]:
            tot[g.type] += 1
            if not any(p.type == g.type and g.iou(p) >= 0.5 for p in preds):
                miss[g.type] += 1
    proc, *_ = run_system_on_docs(eng, gate, test, 0.25)
    ev = evaluate([dict(gold=d["gold"], pred=d["pred"]) for d in proc
                   if d.get("processed") and d.get("gold") is not None])
    dp = delta / len(types); out = {}
    for t in types:
        U = cp_upper(miss[t], tot[t], dp)
        tau = gate.threshold(t)
        out[t] = dict(n=tot[t], phat=round(miss[t] / tot[t], 4), tau=round(tau, 3),
                      certified=round(alpha if tau > 0 else U, 4),
                      realized=round(1 - ev["per_type"].get(t, {}).get("R", 0), 4))
    return out

def multiseed(domain, seeds, engines):
    per = {k: [] for k in engines}; detail = {}
    for seed in seeds:
        stream = gen_domain_corpus(domain, 300, 0.12, seed)
        calib, test = split_stream(stream, 0.4, seed)
        calib_u = [d for d in calib if not d["is_dup"]]
        for name, eng in engines.items():
            gate = fit_gate(eng, calib_u, 0.10)[0] if name == "ours" else None
            proc, n_sup, kinds, dt = run_system_on_docs(
                eng, gate, test, 0.25, with_dedup=(name == "ours"))
            r = summarize(proc, gate, dt); r.update(seed=seed, suppressed=n_sup)
            per[name].append(r)
            if seed == seeds[0]:
                r["F1_ci95"] = bootstrap_ci([dict(gold=d["gold"], pred=d["pred"])
                    for d in proc if d.get("processed") and d.get("gold") is not None])
                detail[name] = r
            print(f"  [{domain}] seed {seed} {name}: F1={r['micro']['F1']:.3f} "
                  f"R={r['micro']['R']:.3f} miss={r['end_to_end_miss_rate']:.3f}", flush=True)
    agg = {n: {m: dict(mean=statistics.mean(x["micro"][m] for x in v),
                       sd=statistics.pstdev(x["micro"][m] for x in v)) for m in ("P","R","F1")}
           for n, v in per.items()}
    for n, v in per.items():
        agg[n]["miss"] = dict(mean=statistics.mean(x["end_to_end_miss_rate"] for x in v),
                              sd=statistics.pstdev(x["end_to_end_miss_rate"] for x in v))
    return dict(aggregate=agg, per_seed=per, seed_first_detail=detail)

def redundancy_sweep(domain, seed):
    out = {}
    for frac in (0.06, 0.12, 0.24):
        stream = gen_domain_corpus(domain, 300, frac, seed)
        dd = Deduplicator(); sup_bytes = tot_bytes = 0; n_sup = fs = 0
        seen_srcs = set()
        for d in stream:
            tot_bytes += len(d["text"])
            dup, _ = dd.check_and_add(d["text"])
            if dup:
                n_sup += 1; sup_bytes += len(d["text"])
                if not d["is_dup"] and d["id"] not in seen_srcs: fs += 1
            if d["is_dup"]: seen_srcs.add(d["src_id"])
        out[str(frac)] = dict(suppressed=n_sup, byte_savings=round(sup_bytes/tot_bytes, 4),
                              false_suppressions=fs, n_docs=len(stream))
    return out

if __name__ == "__main__":
    seeds = (7, 13, 21, 42, 77)
    engines = {"ours": RuleEngine(), "presidio-sm": PresidioEngine("en_core_web_sm"),
               "presidio-lg": PresidioEngine("en_core_web_lg")}
    targets = (("cga-finance", "finance"), ("cga-health", "healthcare"))
    if len(sys.argv) > 1 and sys.argv[1] == "health": targets = targets[1:]
    for name, domain in targets:
        print(f"==== {name} ====", flush=True)
        root = setup_repo(name, domain)
        export_data(root, domain, seeds)
        res = dict(multiseed=multiseed(domain, seeds, engines),
                   bounds=bounds_table(domain, seeds[0]))
        if domain == "finance":
            res["redundancy_sweep"] = redundancy_sweep(domain, seeds[0])
        else:
            res["escalation"] = run_health_escalation(seeds[0])
        json.dump(res, open(f"{root}/results/{domain}_results.json", "w"),
                  indent=1, default=str)
        print(f"saved {root}/results/{domain}_results.json", flush=True)
    print("BUILD COMPLETE", flush=True)
