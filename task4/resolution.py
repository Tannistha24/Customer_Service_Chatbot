from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Optional
# A single, well-known token that means "not tied to a specific region".
# Both None and this string are treated as "global" for comparison purposes.
GLOBAL_REGION = "GLOBAL"

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """A candidate document/chunk that already survived Step 2's
    RBAC + temporal filtering. This mirrors the Step 1 Document/Chunk
    metadata fields that Step 3 actually needs.
    """
    doc_id: str
    product: str
    region: Optional[str]          # None or "GLOBAL" means global policy
    topic_category: str
    effective_date: date
    version: int
    chunk_id: Optional[str] = None
    content: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def normalized_region(self) -> str:
        """Region value used for comparisons. None -> GLOBAL_REGION."""
        if self.region is None:
            return GLOBAL_REGION
        return self.region.strip().upper()

    @property
    def is_global(self) -> bool:
        return self.normalized_region == GLOBAL_REGION

    @property
    def candidate_id(self) -> str:
        """Stable identifier used for lineage/tie-breaking."""
        return f"{self.doc_id}:{self.chunk_id}" if self.chunk_id else self.doc_id


def candidate_from_dict(data: dict) -> Candidate:
    """Helper to build a Candidate from a plain dict, e.g. if Step 1/Step 2
    hand over dicts instead of dataclass instances. Missing required keys
    raise KeyError on purpose - we never invent metadata (Requirement 6).
    """
    return Candidate(
        doc_id=data["doc_id"],
        product=data["product"],
        region=data.get("region"),
        topic_category=data["topic_category"],
        effective_date=data["effective_date"],
        version=data["version"],
        chunk_id=data.get("chunk_id"),
        content=data.get("content", ""),
        metadata=data.get("metadata", {}),
    )


# ---------------------------------------------------------------------------
# Resolution result / audit trail
# ---------------------------------------------------------------------------

@dataclass
class ResolutionDecision:
    """Records what happened to one candidate, for transparency/debugging."""
    candidate_id: str
    kept: bool
    reason: str


@dataclass
class ResolutionResult:
    selected: list          # list[Candidate] that should go to retrieval/generation
    decisions: list         # list[ResolutionDecision] - full audit trail


# ---------------------------------------------------------------------------
# Core resolution logic
# ---------------------------------------------------------------------------

def _safety_belt_temporal_check(candidates: Iterable[Candidate], target_date: date,
                                 decisions: list) -> list:
    """Defensive re-check only. Step 2 already owns full temporal filtering
    (effective_date/expiry windows, historical windows, etc.). This does NOT
    reimplement that logic - it only refuses to select something whose own
    effective_date is after the requested target_date, as a last line of
    defense against a candidate slipping through with inconsistent metadata.
    """
    kept = []
    for c in candidates:
        if c.effective_date > target_date:
            decisions.append(ResolutionDecision(
                candidate_id=c.candidate_id,
                kept=False,
                reason=(f"safety check: effective_date {c.effective_date} is after "
                        f"target_date {target_date} (should have been excluded by Step 2)"),
            ))
            continue
        kept.append(c)
    return kept


def _group_by(candidates: Iterable[Candidate], key_fn):
    groups: dict = {}
    for c in candidates:
        groups.setdefault(key_fn(c), []).append(c)
    return groups


def _pick_best_in_exact_group(group: list, decisions: list) -> Candidate:
    """Given candidates that all share (product, region, topic_category),
    pick exactly one deterministic winner:
        1. latest effective_date wins
        2. tie -> higher version wins
        3. tie -> lowest candidate_id (string) wins, purely for determinism
           (this only matters for literal duplicates with identical
           date+version, which is an edge case but must still be stable)
    """
    ordered = sorted(
        group,
        key=lambda c: (c.effective_date, c.version, c.candidate_id),
        reverse=True,
    )
    winner = ordered[0]
    for loser in ordered[1:]:
        if loser.effective_date != winner.effective_date:
            reason = (f"superseded by {winner.candidate_id}: older effective_date "
                       f"({loser.effective_date} < {winner.effective_date})")
        elif loser.version != winner.version:
            reason = (f"superseded by {winner.candidate_id}: lower version "
                       f"(v{loser.version} < v{winner.version}) at same effective_date")
        else:
            reason = (f"duplicate of {winner.candidate_id}: identical effective_date "
                       f"and version; dropped as redundant")
        decisions.append(ResolutionDecision(loser.candidate_id, kept=False, reason=reason))
    return winner


def _resolve_region_vs_global(group: list, decisions: list) -> list:
    """Given the per-(product, region, topic_category) winners for a single
    (product, topic_category), prefer region-specific policies over the
    global one. Multiple *different* specific regions are not conflicting
    with each other (they apply to different regions) and are all kept.
    """
    specific = [c for c in group if not c.is_global]
    global_ones = [c for c in group if c.is_global]

    if specific:
        for g in global_ones:
            decisions.append(ResolutionDecision(
                candidate_id=g.candidate_id,
                kept=False,
                reason=(f"global policy overridden by region-specific policy "
                        f"({', '.join(sorted(c.normalized_region for c in specific))})"),
            ))
        return specific

    return global_ones


def resolve_conflicts(candidates: Iterable[Candidate], target_date: date) -> ResolutionResult:
    """Main entry point for Step 3.

    Args:
        candidates: the already-authorized, already-temporally-filtered
            candidate set produced by Step 2. Must not be augmented with
            anything outside that set (Requirement 6).
        target_date: the historical/target date already determined by
            Step 2 (e.g. "as of" date for the query). Used only for the
            defensive safety-belt check above - not for primary filtering.

    Returns:
        ResolutionResult with the deterministic, deduplicated selection
        and a full decision audit trail.
    """
    candidates = list(candidates)
    decisions: list = []

    # Safety belt only - Step 2 already did the real temporal filtering.
    candidates = _safety_belt_temporal_check(candidates, target_date, decisions)

    # 1. Group by the exact policy identity: (product, region, topic_category).
    #    Within each such group, collapse duplicates/older versions to one winner.
    exact_groups = _group_by(candidates, lambda c: (c.product, c.normalized_region, c.topic_category))
    per_region_winners = [
        _pick_best_in_exact_group(group, decisions) for group in exact_groups.values()
    ]

    # 2. Group those winners by (product, topic_category) to resolve
    #    region-specific vs. global conflicts for the same underlying policy.
    topic_groups = _group_by(per_region_winners, lambda c: (c.product, c.topic_category))
    final_selection = []
    for group in topic_groups.values():
        final_selection.extend(_resolve_region_vs_global(group, decisions))

    # Record "kept" decisions for the survivors, for a complete audit trail.
    kept_ids = {c.candidate_id for c in final_selection}
    for c in final_selection:
        decisions.append(ResolutionDecision(
            candidate_id=c.candidate_id,
            kept=True,
            reason="selected: no higher-priority candidate for its (product, topic_category)",
        ))

    # Deterministic ordering of the output (not just "some" order from dict iteration).
    final_selection.sort(key=lambda c: (c.product, c.topic_category, c.normalized_region, c.candidate_id))

    return ResolutionResult(selected=final_selection, decisions=decisions)


# ---------------------------------------------------------------------------
# Self-checks / runnable example
# ---------------------------------------------------------------------------

def _make(doc_id, product, region, topic, eff_date, version, chunk_id=None):
    return Candidate(
        doc_id=doc_id,
        product=product,
        region=region,
        topic_category=topic,
        effective_date=eff_date,
        version=version,
        chunk_id=chunk_id,
        content=f"content of {doc_id}",
        metadata={"source": doc_id},
    )


def _run_self_checks():
    target = date(2026, 1, 1)

    # --- Case 1: global vs region-specific conflict -> region-specific wins ---
    c_global = _make("POL-REFUND-GLOBAL", "WidgetPro", None, "refund_policy", date(2025, 1, 1), 1)
    c_region = _make("POL-REFUND-IN", "WidgetPro", "IN", "refund_policy", date(2025, 1, 1), 1)
    result = resolve_conflicts([c_global, c_region], target)
    assert [c.doc_id for c in result.selected] == ["POL-REFUND-IN"], "region-specific should win over global"

    # --- Case 2: older vs newer effective_date -> newer wins ---
    old = _make("POL-DATA-V1", "WidgetPro", "EU", "data_policy", date(2024, 6, 1), 1)
    new = _make("POL-DATA-V2", "WidgetPro", "EU", "data_policy", date(2025, 6, 1), 1)
    result = resolve_conflicts([old, new], target)
    assert [c.doc_id for c in result.selected] == ["POL-DATA-V2"], "newer effective_date should win"

    # --- Case 3: version tie-breaker when effective_date is equal ---
    v1 = _make("POL-SLA-V1", "WidgetPro", "US", "sla", date(2025, 3, 1), 1)
    v2 = _make("POL-SLA-V2", "WidgetPro", "US", "sla", date(2025, 3, 1), 2)
    result = resolve_conflicts([v1, v2], target)
    assert [c.doc_id for c in result.selected] == ["POL-SLA-V2"], "higher version should win on date tie"

    # --- Case 4: duplicate removal (identical date+version) ---
    d1 = _make("POL-DUP-A", "WidgetPro", "US", "warranty", date(2025, 5, 1), 1)
    d2 = _make("POL-DUP-B", "WidgetPro", "US", "warranty", date(2025, 5, 1), 1)
    result = resolve_conflicts([d1, d2], target)
    assert len(result.selected) == 1, "exact duplicates must collapse to a single candidate"

    # --- Case 5: historical query safety belt (future-dated candidate excluded) ---
    future = _make("POL-FUTURE", "WidgetPro", "US", "pricing", date(2030, 1, 1), 1)
    past = _make("POL-PAST", "WidgetPro", "US", "pricing", date(2020, 1, 1), 1)
    result = resolve_conflicts([future, past], target)
    assert [c.doc_id for c in result.selected] == ["POL-PAST"], "future-dated candidate must not be selected"

    # --- Case 6: determinism - same input, run twice, same output ---
    mixed = [c_global, c_region, old, new, v1, v2, d1, d2, future, past]
    r_a = resolve_conflicts(mixed, target)
    r_b = resolve_conflicts(mixed, target)
    assert [c.candidate_id for c in r_a.selected] == [c.candidate_id for c in r_b.selected], \
        "same input must produce identical output"

    # --- Case 7: distinct regions are independent, both survive ---
    us = _make("POL-PRIV-US", "WidgetPro", "US", "privacy", date(2025, 1, 1), 1)
    eu = _make("POL-PRIV-EU", "WidgetPro", "EU", "privacy", date(2025, 1, 1), 1)
    result = resolve_conflicts([us, eu], target)
    assert sorted(c.doc_id for c in result.selected) == ["POL-PRIV-EU", "POL-PRIV-US"], \
        "different regions are not conflicting and should both be kept"

    print("All Step 3 self-checks passed.\n")

    # Small readable demo of the audit trail for the global-vs-region case.
    demo = resolve_conflicts([c_global, c_region], target)
    print("Demo: global vs. region-specific resolution")
    print("Selected:", [c.doc_id for c in demo.selected])
    for d in demo.decisions:
        print(f"  - {d.candidate_id}: kept={d.kept} | {d.reason}")


if __name__ == "__main__":
    _run_self_checks()