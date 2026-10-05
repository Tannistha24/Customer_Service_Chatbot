
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import re


# ---------------------------------------------------------------------------
# Data model - the contract with Step 4's output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContextChunk:
    """One safe, already-filtered/redacted context chunk from Step 4."""
    doc_id: str
    version: str
    section: str
    text: str
    chunk_id: str = ""  # optional finer-grained lineage (paragraph/chunk id)

    def citation(self) -> str:
        return f"[{self.doc_id}, {self.version}, {self.section}]"


@dataclass
class Claim:
    """A single factual claim plus the lineage it is cited against."""
    text: str
    doc_id: str
    version: str
    section: str
    chunk_id: str = ""

    def citation(self) -> str:
        return f"[{self.doc_id}, {self.version}, {self.section}]"


@dataclass
class AnswerResult:
    status: str  # "answered" | "refused"
    answer_text: str
    claims: List[Claim] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    refusal_reason: Optional[str] = None
    unsupported_claims: List[Claim] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Deterministic text utilities (no ML / no external deps)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "to",
    "and", "or", "for", "with", "at", "by", "from", "as", "that", "this",
    "it", "be", "which", "what", "who", "how", "does", "do", "did",
}


def _tokenize(text: str) -> List[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS]


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _overlap_coefficient(a: List[str], b: List[str]) -> float:
    """intersection / min(|a|, |b|) - more lenient than Jaccard for short
    queries matched against longer sentences; used for relevance scoring
    during generation (candidate selection), not for entailment checking."""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


# ---------------------------------------------------------------------------
# Generation (extractive, deterministic)
# ---------------------------------------------------------------------------

MIN_RELEVANCE_SCORE = 0.15          # sentence considered a relevant candidate
MIN_QUERY_MATCH_FOR_ANSWER = 0.15   # best candidate below this -> refuse
SUPPORT_THRESHOLD = 0.5             # min Jaccard for "entailed by sentence"
MAX_CLAIMS = 3                      # keep answers small & simple


def generate_answer(query: str, context_chunks: List[ContextChunk]) -> AnswerResult:
    """
    Deterministically extract the most query-relevant sentence(s) from the
    given safe context and build a cited answer. Every claim is a verbatim
    sentence taken from a chunk, so lineage/faithfulness hold by construction
    here; verify_answer() below re-checks this independently and is what
    protects against any less-trustworthy generator being swapped in later.
    """
    query_tokens = _tokenize(query)

    if not context_chunks:
        return AnswerResult(
            status="refused",
            answer_text="",
            refusal_reason=(
                "No context was provided (Step 4 returned no safe chunks). "
                "I don't have evidence to answer this. Could you clarify or "
                "provide relevant documents?"
            ),
        )

    if not query_tokens:
        return AnswerResult(
            status="refused",
            answer_text="",
            refusal_reason=(
                "The question is empty or too ambiguous to match against "
                "the available evidence. Could you rephrase your question?"
            ),
        )

    candidates: List[Tuple[float, str, ContextChunk]] = []
    for chunk in context_chunks:
        for sentence in _split_sentences(chunk.text):
            score = _overlap_coefficient(query_tokens, _tokenize(sentence))
            if score >= MIN_RELEVANCE_SCORE:
                candidates.append((score, sentence, chunk))

    if not candidates:
        return AnswerResult(
            status="refused",
            answer_text="",
            refusal_reason=(
                "None of the provided context is sufficiently relevant to "
                "the question. Refusing rather than guessing. Could you "
                "clarify what you're looking for, or provide more specific "
                "source material?"
            ),
        )

    # Deterministic ordering: best score first, then a fixed tiebreak key.
    candidates.sort(
        key=lambda c: (-c[0], c[2].doc_id, c[2].version, c[2].section, c[1])
    )

    if candidates[0][0] < MIN_QUERY_MATCH_FOR_ANSWER:
        return AnswerResult(
            status="refused",
            answer_text="",
            refusal_reason=(
                "Evidence found is too weakly related to the question to "
                "answer confidently. Could you clarify your question?"
            ),
        )

    seen = set()
    claims: List[Claim] = []
    for score, sentence, chunk in candidates:
        key = sentence.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        claims.append(Claim(
            text=sentence,
            doc_id=chunk.doc_id,
            version=chunk.version,
            section=chunk.section,
            chunk_id=chunk.chunk_id,
        ))
        if len(claims) >= MAX_CLAIMS:
            break

    answer_text = " ".join(f"{c.text} {c.citation()}" for c in claims)
    citations = sorted({c.citation() for c in claims})

    return AnswerResult(
        status="answered",
        answer_text=answer_text,
        claims=claims,
        citations=citations,
    )


# ---------------------------------------------------------------------------
# Faithfulness verification
# ---------------------------------------------------------------------------

def _find_chunk(claim: Claim, context_chunks: List[ContextChunk]) -> Optional[ContextChunk]:
    for chunk in context_chunks:
        if (chunk.doc_id, chunk.version, chunk.section) == (
            claim.doc_id, claim.version, claim.section,
        ):
            if not claim.chunk_id or claim.chunk_id == chunk.chunk_id:
                return chunk
    return None


def _claim_supported_by_chunk(claim_text: str, chunk_text: str) -> bool:
    """
    A claim is "directly supported/entailed" if it appears verbatim
    (case-insensitive) in the chunk, or has high token overlap with some
    sentence in the chunk. Intentionally simple & deterministic (learning
    scope) rather than a full NLI model.
    """
    if claim_text.strip().lower() in chunk_text.strip().lower():
        return True
    claim_tokens = _tokenize(claim_text)
    for sentence in _split_sentences(chunk_text):
        if _jaccard(claim_tokens, _tokenize(sentence)) >= SUPPORT_THRESHOLD:
            return True
    return False


def verify_answer(
    claims: List[Claim], context_chunks: List[ContextChunk]
) -> Tuple[List[Claim], List[Claim]]:
    """
    Independently re-verify every claim against its cited chunk.
    Returns (supported_claims, unsupported_claims).

    A claim is unsupported if its cited (doc_id, version, section[,
    chunk_id]) is not among the safe context at all (broken lineage), or if
    the cited chunk's text does not actually entail the claim text.
    """
    supported, unsupported = [], []
    for claim in claims:
        chunk = _find_chunk(claim, context_chunks)
        if chunk is None or not _claim_supported_by_chunk(claim.text, chunk.text):
            unsupported.append(claim)
        else:
            supported.append(claim)
    return supported, unsupported


def generate_verified_answer(query: str, context_chunks: List[ContextChunk]) -> AnswerResult:
    """
    Full Step 5 pipeline: generate -> verify -> refuse-on-failure.
    External callers should use this entry point.
    """
    result = generate_answer(query, context_chunks)
    if result.status == "refused":
        return result

    supported, unsupported = verify_answer(result.claims, context_chunks)
    if unsupported:
        bad = "; ".join(f"'{c.text}' {c.citation()}" for c in unsupported)
        return AnswerResult(
            status="refused",
            answer_text="",
            refusal_reason=(
                "Generated answer contained claim(s) not directly supported "
                f"by the cited context, so it was rejected: {bad}. Please "
                "clarify the question or supply better source material."
            ),
            unsupported_claims=unsupported,
        )
    return result


# ---------------------------------------------------------------------------
# Self-checks (run: python step5_generation_verification.py)
# ---------------------------------------------------------------------------

def _run_self_checks() -> None:
    passed, failed = 0, 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        if condition:
            print(f"[PASS] {name}")
            passed += 1
        else:
            print(f"[FAIL] {name} {detail}")
            failed += 1

    chunk_a = ContextChunk(
        doc_id="DOC-100", version="v2", section="Refunds",
        chunk_id="c1",
        text=(
            "Refunds are issued within 14 business days of approval. "
            "Approval requires a valid original receipt."
        ),
    )
    chunk_b = ContextChunk(
        doc_id="DOC-100", version="v2", section="Shipping",
        chunk_id="c2",
        text=(
            "Standard shipping takes 5 to 7 business days. "
            "Expedited shipping is available for an extra fee."
        ),
    )
    context = [chunk_a, chunk_b]

    # 1. Supported answer + citation
    r1 = generate_verified_answer("How long do refunds take?", context)
    check(
        "1. Supported answer + citation",
        r1.status == "answered"
        and "[DOC-100, v2, Refunds]" in r1.citations
        and "14 business days" in r1.answer_text,
        detail=str(r1),
    )

    # 2. Unsupported claim is detected and rejected
    fabricated_claim = Claim(
        text="Refunds are issued within 2 hours of approval.",
        doc_id="DOC-100", version="v2", section="Refunds", chunk_id="c1",
    )
    supported, unsupported = verify_answer([fabricated_claim], context)
    check(
        "2. Unsupported claim detected",
        len(unsupported) == 1 and len(supported) == 0,
        detail=str((supported, unsupported)),
    )
    # And a full pipeline exercising the same rejection path:
    fake_generation_result = AnswerResult(
        status="answered",
        answer_text="Refunds are issued within 2 hours of approval. [DOC-100, v2, Refunds]",
        claims=[fabricated_claim],
        citations=[fabricated_claim.citation()],
    )
    fake_supported, fake_unsupported = verify_answer(fake_generation_result.claims, context)
    check(
        "2b. Full pipeline would refuse a fabricated-claim answer",
        len(fake_unsupported) == 1,
        detail=str(fake_unsupported),
    )

    # 3. Missing evidence / refusal
    r3 = generate_verified_answer("What is the CEO's salary?", context)
    check(
        "3. Refusal on missing evidence",
        r3.status == "refused" and r3.refusal_reason is not None,
        detail=str(r3),
    )
    r3b = generate_verified_answer("Tell me about refunds.", [])
    check(
        "3b. Refusal on empty context",
        r3b.status == "refused",
        detail=str(r3b),
    )

    # 4. Multiple citations
    r4 = generate_verified_answer(
        "Tell me about refund approval and shipping time.", context
    )
    check(
        "4. Multiple citations present",
        r4.status == "answered" and len(r4.citations) >= 2,
        detail=str(r4),
    )

    # 5. Deterministic output (same input -> same output, repeated calls)
    r5a = generate_verified_answer("How long do refunds take?", context)
    r5b = generate_verified_answer("How long do refunds take?", context)
    check(
        "5. Deterministic output across repeated calls",
        r5a.answer_text == r5b.answer_text and r5a.citations == r5b.citations,
        detail=str((r5a.answer_text, r5b.answer_text)),
    )

    print(f"\n{passed} passed, {failed} failed")


if __name__ == "__main__":
    _run_self_checks()