"""The conformally gated agentic pipeline as a LangGraph state machine.
Nodes: ingest -> dedup -> detect -> gate -> (review?) -> anonymize -> persist.
Every node appends to an audit trail keyed by content hash."""
from __future__ import annotations
import hashlib, time
from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from .core import Span, ConformalGate, Deduplicator, Anonymizer, canon

class PipelineState(TypedDict, total=False):
    doc_id: str; text: str; content_hash: str
    suppressed: bool; dup_kind: str
    spans: list; accepted: list; review: list; rejected: list
    anonymized: str; audit: list

def build_pipeline(engine, gate: ConformalGate, dedup: Deduplicator,
                   anonymizer: Anonymizer, review_floor: float = 0.25,
                   review_policy: str = "accept"):
    def _audit(state, node, **info):
        state.setdefault("audit", []).append(
            dict(node=node, hash=state.get("content_hash", ""), t=time.time(), **info))

    def ingest(state: PipelineState):
        state["content_hash"] = hashlib.sha256(canon(state["text"]).encode()).hexdigest()
        _audit(state, "ingest"); return state

    def dedup_node(state: PipelineState):
        is_dup, kind = dedup.check_and_add(state["text"])
        state["suppressed"], state["dup_kind"] = is_dup, kind
        _audit(state, "dedup", suppressed=is_dup, kind=kind); return state

    def detect(state: PipelineState):
        state["spans"] = engine.detect(state["text"])
        _audit(state, "detect", n=len(state["spans"])); return state

    def gate_node(state: PipelineState):
        acc, rev, rej = [], [], []
        for sp in state["spans"]:
            d = gate.decide(sp, review_floor)
            (acc if d == "accept" else rev if d == "review" else rej).append(sp)
        state["accepted"], state["review"], state["rejected"] = acc, rev, rej
        _audit(state, "gate", accepted=len(acc), review=len(rev), rejected=len(rej))
        return state

    def review_node(state: PipelineState):
        if review_policy == "accept":
            state["accepted"] = state["accepted"] + state["review"]
        _audit(state, "review", policy=review_policy, n=len(state["review"]))
        state["review_done"] = True
        return state

    def anonymize(state: PipelineState):
        state["anonymized"] = anonymizer.apply(state["text"], state["accepted"])
        _audit(state, "anonymize"); return state

    def persist(state: PipelineState):
        _audit(state, "persist"); return state

    g = StateGraph(PipelineState)
    for name, fn in [("ingest", ingest), ("dedup", dedup_node), ("detect", detect),
                     ("gate", gate_node), ("review", review_node),
                     ("anonymize", anonymize), ("persist", persist)]:
        g.add_node(name, fn)
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "dedup")
    g.add_conditional_edges("dedup",
        lambda s: "suppressed" if s.get("suppressed") else "detect",
        {"suppressed": END, "detect": "detect"})
    g.add_edge("detect", "gate")
    g.add_conditional_edges("gate",
        lambda s: "review" if s.get("review") else "anonymize",
        {"review": "review", "anonymize": "anonymize"})
    g.add_edge("review", "anonymize")
    g.add_edge("anonymize", "persist")
    g.add_edge("persist", END)
    return g.compile()
