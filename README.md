# finance_paper — reproducible artifact
Paper: paper/finance_paper.tex (+ compiled PDF). Every number traces to results/finance_results.json.
Layout: synpii/ (SDK) · data/ (five-seed corpora, gold spans) · results/ · logs/ · paper/ (tex, IEEEtran.cls, architecture figure + standalone source).
Reproduce: pip install presidio-analyzer==2.2.364 datasketch langgraph; install spaCy en_core_web_sm and en_core_web_lg; then: python run_all.py
Companion repos share infrastructure (disclosed in the paper): base framework (cga), and the other domain study.
Certified-bound step: python e2e_bound.py. LLM-engine harness: llm_engine_eval.html.
