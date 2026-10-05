import os
import re
import json
from datetime import datetime,timezone

DEFAULT_MIN_ACCEPTABLE_SCORE = 0.35   # absolute floor a version must clear
DEFAULT_MAX_ALLOWED_DROP = 0.05       # how much worse than baseline is tolerated

MIN_ACCEPTABLE_SCORE = DEFAULT_MIN_ACCEPTABLE_SCORE
MAX_ALLOWED_DROP = DEFAULT_MAX_ALLOWED_DROP

StopWords={
    "the","a", "an", "is", "are", "was", "were", "do", "does", "did",
    "how", "what", "when", "where", "why", "who", "can", "i", "you",
    "to", "of", "for", "in", "on", "and", "or", "my", "your", "it",
    "this", "that", "with", "be", "have", "has", "please",
}
def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

def load_eval_config(path="eval_config.json"):
    """
    Load threshold settings from a small JSON config file, creating a
    default one on first run so it's easy for a beginner to find and
    tweak later. This is the recommended place to change thresholds -
    no code editing required.
    """
    if not os.path.exists(path):
        defaults={
            "min_acceptable_score":DEFAULT_MIN_ACCEPTABLE_SCORE,
            "max_allowed_drop":DEFAULT_MAX_ALLOWED_DROP,
        }
        with open(path,"w",encoding="utf-8")as f:
            json.dump(defaults,f,indent=2)
        print(f"[eval_config] No config found - created defaults at: {path}")
        return defaults
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
 
    # Fill in anything missing rather than crashing on a partial file.
    config.setdefault("min_acceptable_score", DEFAULT_MIN_ACCEPTABLE_SCORE)
    config.setdefault("max_allowed_drop", DEFAULT_MAX_ALLOWED_DROP)
    return config
 
 
# --------------------------------------------------------------------
# 1. Evaluation dataset (the "answer key")
# --------------------------------------------------------------------
 
DEFAULT_EVAL_DATASET = [
    {"question": "What are your business hours?",
     "expected_answer": "We are open Monday to Friday, 9 AM to 6 PM."},
    {"question": "How do I reset my password?",
     "expected_answer": "Go to account settings and click 'Reset Password'."},
    {"question": "What is your return policy?",
     "expected_answer": "You can return items within 30 days of purchase."},
    {"question": "How can I track my order?",
     "expected_answer": "Use the tracking link sent to your email after checkout."},
    {"question": "Do you offer refunds?",
     "expected_answer": "Yes, refunds are issued within 5-7 business days."},
    {"question": "How do I contact customer support?",
     "expected_answer": "You can email support@example.com or use live chat."},
    {"question": "Can I change my shipping address after ordering?",
     "expected_answer": "Contact support within 1 hour of ordering to change the address."},
    {"question": "What payment methods do you accept?",
     "expected_answer": "We accept credit cards, debit cards, and PayPal."},
    {"question": "How do I cancel my subscription?",
     "expected_answer": "Go to account settings and select 'Cancel Subscription'."},
    {"question": "Is international shipping available?",
     "expected_answer": "Yes, we ship to most countries with additional shipping fees."},
    {"question": "How long does delivery take?",
     "expected_answer": "Standard delivery takes 3-5 business days."},
    {"question": "Can I get an invoice for my purchase?",
     "expected_answer": "Invoices are automatically emailed after every purchase."},
    {"question": "What should I do if I received a damaged item?",
     "expected_answer": "Contact support with photos of the damage for a replacement."},
    {"question": "Do you have a loyalty or rewards program?",
     "expected_answer": "Yes, you earn points on every purchase through our rewards program."},
    {"question": "How do I update my email address?",
     "expected_answer": "Go to account settings and edit your profile information."},
]
 
 
def validate_eval_dataset(dataset, source_description="eval dataset"):
    """
    Check that the evaluation dataset is well-formed before using it.
    Raises a clear ValueError pointing at exactly which entry is wrong,
    instead of letting a bad entry cause a confusing crash later on.
    """
    if not isinstance(dataset, list) or len(dataset) == 0:
        raise ValueError(
            f"{source_description} must be a non-empty list of "
            f"{{'question': ..., 'expected_answer': ...}} objects."
        )
 
    for index, entry in enumerate(dataset):
        if not isinstance(entry, dict):
            raise ValueError(
                f"{source_description}: entry #{index + 1} is not an object "
                f"(got {type(entry).__name__})."
            )
 
        question = entry.get("question")
        expected_answer = entry.get("expected_answer")
 
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"{source_description}: entry #{index + 1} is missing a "
                f"non-empty 'question' field."
            )
 
        if not isinstance(expected_answer, str) or not expected_answer.strip():
            raise ValueError(
                f"{source_description}: entry #{index + 1} (question: "
                f"'{question}') is missing a non-empty 'expected_answer' field."
            )
 
    # Soft guidance only - not a hard error, since a slightly smaller or
    # larger set shouldn't stop the pipeline from running.
    if len(dataset) < 10 or len(dataset) > 20:
        print(
            f"[eval_dataset] Warning: {source_description} has "
            f"{len(dataset)} questions - the recommended range is 10-20."
        )
 
 
def ensure_eval_dataset(path):
    """
    Load the evaluation dataset from `path`. If it doesn't exist yet,
    create it with a default starter set so beginners have something
    to run immediately, then edit later to match their real product.
    Either way, the dataset is validated before use.
    """
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        validate_eval_dataset(dataset, source_description=f"'{path}'")
        return dataset
 
    with open(path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_EVAL_DATASET, f, indent=2)
    print(f"[eval_dataset] No dataset found - created a starter one at: {path}")
    validate_eval_dataset(DEFAULT_EVAL_DATASET, source_description="default eval dataset")
    return DEFAULT_EVAL_DATASET
 
 
# --------------------------------------------------------------------
# 2. Simple text handling helpers (no external libraries)
# --------------------------------------------------------------------
 
def to_word_set(text):
    """Lowercase, strip punctuation, split into a set of words."""
    words = re.findall(r"[a-z0-9']+", text.lower())
    return set(words)
 
 
def extract_keywords(text, min_len=4):
    """
    Pull out the 'important' words from a piece of text - used to
    check whether an answer is actually grounded in retrieved content.
    """
    words = to_word_set(text)
    return {w for w in words if len(w) >= min_len and w not in StopWords}
 
 
def jaccard_similarity(set_a, set_b):
    """Simple overlap score between 0 and 1. No shared words = 0."""
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)
 
 
# --------------------------------------------------------------------
# 3. Loading KB content and doing simple retrieval
# --------------------------------------------------------------------
 
def load_kb_chunks(kb_data_path):
    """
    Read every text-like file inside a KB version's kb_data/ folder and
    split it into small chunks (paragraphs). This is written to never
    raise an exception - a missing folder, empty folder, or unreadable
    file should safely result in fewer (or zero) chunks, not a crash.
    """
    chunks = []
 
    if not kb_data_path or not os.path.isdir(kb_data_path):
        print(f"[quality_eval] Warning: KB data folder not found: {kb_data_path}")
        return chunks
 
    for root, _dirs, files in os.walk(kb_data_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                # Skip anything unreadable (binary, permission issue,
                # odd encoding, etc.) rather than stopping the whole
                # evaluation over one bad file.
                print(f"[quality_eval] Skipping unreadable file '{file_path}': {e}")
                continue
 
            for paragraph in content.split("\n\n"):
                paragraph = paragraph.strip()
                if paragraph:
                    chunks.append(paragraph)
 
    if not chunks:
        print(f"[quality_eval] Warning: no readable content found in: {kb_data_path}")
 
    return chunks
 
 
def retrieve_best_chunk(question, chunks):
    """
    Find the chunk with the highest word-overlap with the question.
    Returns (best_chunk, retrieval_score). If there are no chunks at
    all, returns (None, 0.0).
    """
    if not chunks:
        return None, 0.0
 
    question_words = to_word_set(question)
    best_chunk = None
    best_score = -1.0
 
    for chunk in chunks:
        score = jaccard_similarity(question_words, to_word_set(chunk))
        if score > best_score:
            best_score = score
            best_chunk = chunk
 
    return best_chunk, best_score
 
 
# --------------------------------------------------------------------
# 4. Scoring one question, then a whole version
# --------------------------------------------------------------------
 
def evaluate_question(qa_pair, chunks, min_acceptable_score):
    """
    Test a single question/expected-answer pair against the KB chunks.
    Returns a small dict with the retrieved chunk, its scores, and a
    plain-language failure reason when the question scored poorly.
    """
    question = qa_pair["question"]
    expected_answer = qa_pair["expected_answer"]
 
    retrieved_chunk, retrieval_score = retrieve_best_chunk(question, chunks)
    retrieved_text = retrieved_chunk or ""
 
    # Similarity: does the retrieved content overlap with the expected answer?
    similarity_score = jaccard_similarity(
        to_word_set(expected_answer), to_word_set(retrieved_text)
    )
 
    # Grounding: do the important keywords from the expected answer
    # actually appear somewhere in the retrieved content?
    expected_keywords = extract_keywords(expected_answer)
    if expected_keywords:
        found = expected_keywords & to_word_set(retrieved_text)
        grounding_score = len(found) / len(expected_keywords)
    else:
        grounding_score = 0.0
 
    question_score = (similarity_score + grounding_score) / 2
 
    # Plain-language reason for a low score - useful when reading the
    # report later without having to re-derive it from raw numbers.
    failure_reason = None
    if not chunks:
        failure_reason = "No KB content was available to search."
    elif not retrieved_text:
        failure_reason = "No matching content was found in the KB."
    elif question_score < min_acceptable_score:
        if similarity_score < grounding_score:
            failure_reason = "Retrieved content doesn't closely match the expected answer."
        else:
            failure_reason = "Retrieved content is missing key facts from the expected answer."
 
    return {
        "question": question,
        "expected_answer": expected_answer,
        "retrieved_chunk": retrieved_text,
        "retrieval_score": round(retrieval_score, 3),
        "similarity_score": round(similarity_score, 3),
        "grounding_score": round(grounding_score, 3),
        "question_score": round(question_score, 3),
        "failure_reason": failure_reason,
    }
 
 
def evaluate_kb_version(kb_data_path, eval_dataset, min_acceptable_score):
    """
    Run every question in eval_dataset against one KB version's data.
    Returns a report dict with per-question detail and overall scores.
    Safe to call even if kb_data_path is missing or empty - it will
    simply score very low instead of raising an exception.
    """
    chunks = load_kb_chunks(kb_data_path)
 
    per_question_results = [
        evaluate_question(qa, chunks, min_acceptable_score) for qa in eval_dataset
    ]
 
    def avg(key):
        values = [r[key] for r in per_question_results]
        return round(sum(values) / len(values), 3) if values else 0.0
 
    return {
        "num_questions": len(per_question_results),
        "num_kb_chunks": len(chunks),
        "avg_similarity_score": avg("similarity_score"),
        "avg_grounding_score": avg("grounding_score"),
        "overall_score": avg("question_score"),
        "per_question_results": per_question_results,
    }
 
 
# --------------------------------------------------------------------
# 5. Baseline tracking and approval bookkeeping
# --------------------------------------------------------------------
 
def load_baseline(eval_results_dir):
    path = os.path.join(eval_results_dir, "baseline.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
 
 
def save_baseline(eval_results_dir, version_name, overall_score):
    path = os.path.join(eval_results_dir, "baseline.json")
    data = {
        "version": version_name,
        "overall_score": overall_score,
        "approved_at": utc_now_iso(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
 
 
def append_to_approved_log(eval_results_dir, version_name, overall_score):
    path = os.path.join(eval_results_dir, "approved_versions.json")
    history = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            history = json.load(f)
 
    history.append({
        "version": version_name,
        "overall_score": overall_score,
        "approved_at": utc_now_iso(),
    })
 
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
 
 
def raise_alert(eval_results_dir, message):
    """
    A minimal alert: print it loudly and append it to a plain text log.
    Full alerting/monitoring belongs to a later step - this is just
    enough for the quality gate to make a rejection visible.
    """
    print(f"\n*** QUALITY GATE ALERT ***\n{message}\n")
    log_path = os.path.join(eval_results_dir, "alerts.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{utc_now_iso()}] {message}\n")
 
 
# --------------------------------------------------------------------
# 6. The main entry point: run the gate for one KB version
# --------------------------------------------------------------------
 
def run_quality_gate(
    version_name,
    kb_data_path,
    eval_dataset_path="eval_dataset.json",
    eval_results_dir="eval_results",
    eval_config_path="eval_config.json",
    min_acceptable_score=None,
    max_allowed_drop=None,
):
    """
    Evaluate a KB version and decide APPROVED/REJECTED.
 
    version_name:          name of the version being tested (e.g. from Step 2)
    kb_data_path:           path to that version's kb_data/ folder (read-only).
                            Safe to pass a missing/empty path - it will not crash.
    eval_dataset_path:      path to the question/answer test file
    eval_results_dir:       where reports, baseline, and alerts are stored
    eval_config_path:       path to the thresholds config file
    min_acceptable_score:   optional override for this run only (skips config file)
    max_allowed_drop:       optional override for this run only (skips config file)
 
    Returns a dict describing the decision. Does NOT activate anything.
    """
    os.makedirs(eval_results_dir, exist_ok=True)
 
    # Thresholds: explicit function arguments win, otherwise fall back
    # to the config file (which is the easy, no-code way to tune them).
    config = load_eval_config(eval_config_path)
    if min_acceptable_score is None:
        min_acceptable_score = config["min_acceptable_score"]
    if max_allowed_drop is None:
        max_allowed_drop = config["max_allowed_drop"]
 
    eval_dataset = ensure_eval_dataset(eval_dataset_path)
    report = evaluate_kb_version(kb_data_path, eval_dataset, min_acceptable_score)
 
    baseline = load_baseline(eval_results_dir)
    if baseline is None:
        pass_threshold = min_acceptable_score
        baseline_info = None
    else:
        pass_threshold = max(
            min_acceptable_score, baseline["overall_score"] - max_allowed_drop
        )
        baseline_info = baseline
 
    overall_score = report["overall_score"]
    passed = overall_score >= pass_threshold
    decision = "APPROVED" if passed else "REJECTED"
 
    result = {
        "version": version_name,
        "evaluated_at": utc_now_iso(),
        "overall_score": overall_score,
        "avg_similarity_score": report["avg_similarity_score"],
        "avg_grounding_score": report["avg_grounding_score"],
        "num_questions": report["num_questions"],
        "num_kb_chunks": report["num_kb_chunks"],
        "min_acceptable_score_used": round(min_acceptable_score, 3),
        "max_allowed_drop_used": round(max_allowed_drop, 3),
        "pass_threshold_used": round(pass_threshold, 3),
        "baseline": baseline_info,
        "passed": passed,
        "decision": decision,
        "per_question_results": report["per_question_results"],
    }
 
    # Save the full report for this version, regardless of outcome.
    report_path = os.path.join(eval_results_dir, f"{version_name}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
 
    if passed:
        save_baseline(eval_results_dir, version_name, overall_score)
        append_to_approved_log(eval_results_dir, version_name, overall_score)
        print(
            f"[quality_gate] APPROVED - version '{version_name}' "
            f"scored {overall_score} (threshold: {round(pass_threshold, 3)})"
        )
    else:
        raise_alert(
            eval_results_dir,
            f"Version '{version_name}' REJECTED - scored {overall_score}, "
            f"needed at least {round(pass_threshold, 3)}.",
        )
 
    return result
 
 
if __name__ == "__main__":
    pass