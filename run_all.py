#!/usr/bin/env python3
"""One-command reproduction of every number in the paper.
Usage: pip install presidio-analyzer datasketch langgraph && spaCy models sm+lg, then: python run_all.py"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import build_domain_papers as B
if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    seeds = (7, 13, 21, 42, 77)
    engines = {"ours": B.RuleEngine(), "presidio-sm": B.PresidioEngine("en_core_web_sm"),
               "presidio-lg": B.PresidioEngine("en_core_web_lg")}
    res = dict(multiseed=B.multiseed("finance", seeds, engines), bounds=B.bounds_table("finance", 7))
    if "finance" == "finance": res["redundancy_sweep"] = B.redundancy_sweep("finance", 7)
    else: res["escalation"] = B.run_health_escalation(7)
    json.dump(res, open(f"{root}/results/finance_results.json", "w"), indent=1, default=str)
    print("done -> results/finance_results.json")
