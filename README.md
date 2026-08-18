Conformal PII Gating for Financial Documents

Reproducible artifact for "Conformal PII Gating for Financial Documents" (IEEE format, `paper/finance_paper.tex` + compiled `paper/finance_paper.pdf`). Every number in the paper traces back to a key in `results/finance_results.json`.
Financial document streams — KYC files, wire memos, loan notes — carry PII whose dominant identifiers are checksummed: card numbers validate under Luhn, IBANs under ISO 7064 mod-97. This project shows that checksum anchoring turns split conformal calibration into an unusually strong control for redaction:
Verified identifiers produce near-separable detector scores, so per-type conformal thresholds settle in `[0.88, 0.99]` and carry a finite-sample certificate that at most a user-chosen `α` of true spans leak end to end.
Person names — the one type with no checksum — are handled by the same framework's measured regime: an exact binomial (Clopper–Pearson) upper bound on the candidate-miss rate.
A gate ablation quantifies what conformal calibration replaces: a fixed threshold of `0.85` either routes 3.05 spans/doc to human review, or, if that queue is dropped, silently misses 21.6% of PII at precision 1.000 — while the calibrated gate holds end-to-end miss at 3.2% under either review policy.
Content-addressed + MinHash deduplication precedes detection; at injected redundancy of 6/12/24% it recovers byte savings of 6.0/12.2/24.0% with zero false suppressions, and accepted spans are replaced by surrogates regenerated checksum-valid so downstream format validators keep passing.
Headline results (5-seed synthetic corpus, `results/finance_results.json`)
System	Precision	Recall	F1	End-to-end miss
Ours (rule engine + conformal gate)	0.871	0.967	0.916	3.3%
Presidio (`en_core_web_sm`)	0.545	0.690	0.609	31.0%
Presidio (`en_core_web_lg`)	0.691	0.703	0.697	29.7%
Mean ± population sd over seeds `{7, 13, 21, 42, 77}`; see `multiseed.aggregate` in the results file for the standard deviations and `multiseed.per_seed` for the raw per-seed rows.
Certified vs. realized per-type leakage (seed 7, `α = 0.10`, `δ = 0.10`)
Checksummed types (`ACCOUNT_NUMBER`, `ADDRESS`, `CREDIT_CARD`, `DOB`, `EMAIL`, `IBAN`, ...) calibrate to `τ ∈ [0.88, 0.99]` with realized test-set miss of `0.0`. `PERSON` — the unchecksummed type — collapses to `τ = 0`, falling back to the Clopper–Pearson bound (certified `0.257`, realized `0.150`). Full per-type table in `bounds`, reproduced independently by `python e2e_bound.py` → `results/e2e_bounds.json`.
Honest external validation (out of scope for the synthetic-corpus claims above)
Run against two third-party HuggingFace corpora (`data/external/`) to test transfer beyond our own templates — reported as a limitation, not a win:
Corpus	Ours F1	Presidio F1	Ours end-to-end miss
Gretel finance (n=3,000)	0.164	0.291	49.2%
Nemotron finance (n=2,394)	0.145	0.258	44.9%
On documents never templated against, roughly half of the rule engine's checksummed-type thresholds collapse to `τ = 0` and miss rises sharply — see `results/external_gretel_finance.json`, `results/external_nemotron_finance.json`, and the paper's "External Validation" and "Limitations" sections for the full discussion.

Repository layout
```text
synpii/                  Core SDK: taxonomy, split-conformal gate, dedup, surrogate anonymizer, metrics
  core.py                Span/IoU, ConformalGate (Algorithm 1), Deduplicator, Anonymizer, evaluate()
  engines.py              detect(text) -> [Span] backends: rule engine (checksums + context), Presidio, LLM
  generator.py            SynPII-FH synthetic corpus generator (finance + clinical templates, 11 entity types)
  external_datasets.py     HuggingFace loaders (Gretel, Nemotron) mapped onto the canonical taxonomy
  piibench.py              Official PIIBench evaluation protocol adapter
  graph.py                 LangGraph pipeline wiring for an auditable end-to-end run
data/                    Five-seed synthetic corpora (seeds 7/13/21/42/77) with gold spans, + external/ corpora
results/                 All experiment outputs as JSON; every paper number cites a key here
logs/                    Run logs
paper/                   finance_paper.tex, IEEEtran.cls, architecture figure (+ standalone source), compiled PDF
run_all.py               One-command reproduction of every number in the paper
run_experiments.py       Experiment harness: synthetic / ablations / dedup-stress / piibench / external / graph steps
e2e_bound.py             Standalone certified end-to-end leakage bound (Corollary 1)
build_domain_papers.py   Shared builder for this repo and its companion healthcare-domain study
```
Reproducing the results
```bash
pip install presidio-analyzer==2.2.364 datasketch langgraph
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg

python run_all.py          # writes results/finance_results.json
python e2e_bound.py        # certified end-to-end bound -> results/e2e_bounds.json
```
Individual experiment steps are also available directly via the harness:
```bash
python run_experiments.py synthetic ablations dedupstress graph
python run_experiments.py external                                  # requires HF datasets access
python run_experiments.py piibench --test-file <path> --calib-file <path>
```
Companion work
This repository shares its `synpii/` SDK and experiment harness with a base framework repo (CGA) and a companion domain study on healthcare documents, as disclosed in the paper.
