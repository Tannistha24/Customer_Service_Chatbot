import re
from typing import Callable, Dict, List


# Regex patterns for common types of personally identifiable information.
PII_PATTERNS = [
    (
        "email",
        re.compile(
            r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
            re.IGNORECASE,
        ),
        "[EMAIL_REDACTED]",
    ),
    (
        "credit_card",
        re.compile(
            r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"
        ),
        "[CARD_REDACTED]",
    ),
    (
        "ssn",
        re.compile(
            r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"
        ),
        "[SSN_REDACTED]",
    ),
    (
        "phone",
        re.compile(
            r"(?<!\w)"
            r"(?:\+?\d{1,3}[-.\s]?)?"
            r"(?:\(?\d{2,5}\)?[-.\s]?)?"
            r"\d{3,5}[-.\s]?\d{3,5}"
            r"(?!\w)"
        ),
        "[PHONE_REDACTED]",
    ),
    (
        "date_of_birth",
        re.compile(
            r"\b(?:date of birth|dob)\s*[:=-]\s*"
            r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            re.IGNORECASE,
        ),
        "[DOB_REDACTED]",
    ),
    (
        "name",
        re.compile(
            r"\b(?:full name|first name|last name)\s*[:=-]\s*"
            r"[A-Za-z][A-Za-z .'-]{1,60}(?=[,.;\n]|$)",
            re.IGNORECASE,
        ),
        "[NAME_REDACTED]",
    ),
    (
        "address",
        re.compile(
            r"\b(?:home address|street address|address)\s*[:=-]\s*"
            r"[^,;\n]+",
            re.IGNORECASE,
        ),
        "[ADDRESS_REDACTED]",
    ),
]


# Simple heuristic rules for detecting prompt injection.
INJECTION_RULES = [
    (
        "instruction_override",
        re.compile(
            r"\b(ignore|disregard|forget|override|do not follow)\b"
            r".{0,60}"
            r"\b(previous|prior|earlier|system|developer|all)\b"
            r".{0,30}"
            r"\b(instruction|message|rule|prompt)s?\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "attempts to override earlier instructions",
    ),
    (
        "secret_prompt_request",
        re.compile(
            r"\b(reveal|show|print|tell me|expose|repeat)\b"
            r".{0,50}"
            r"\b(system prompt|hidden prompt|secret instructions|developer message)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "requests hidden system or developer instructions",
    ),
    (
        "role_impersonation",
        re.compile(
            r"\b(act as|pretend to be|you are now|roleplay as)\b"
            r".{0,40}"
            r"\b(system|developer|admin|root|unrestricted)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "tries to impersonate a privileged role",
    ),
    (
        "safety_bypass",
        re.compile(
            r"\b(jailbreak|bypass|disable|evade|ignore)\b"
            r".{0,40}"
            r"\b(safety|security|guardrail|filter|restriction)s?\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "tries to bypass a safety or security guardrail",
    ),
]


def mask_pii(text: str) -> Dict[str, object]:
    """
    Replace detected PII with safe placeholder text.

    Returns:
        A dictionary containing the sanitized text and detected PII types.
    """
    sanitized_text = text
    found_types: List[str] = []

    # Apply every PII pattern to the input.
    for pii_type, pattern, replacement in PII_PATTERNS:
        sanitized_text, replacement_count = pattern.subn(
            replacement,
            sanitized_text,
        )

        if replacement_count > 0:
            found_types.append(pii_type)

    return {
        "text": sanitized_text,
        "found_types": found_types,
    }


def detect_prompt_injection(text: str) -> List[Dict[str, str]]:
    """
    Check the input against prompt-injection heuristic rules.

    Returns:
        A list of rules that matched the input.
    """
    matches: List[Dict[str, str]] = []

    for rule_name, pattern, explanation in INJECTION_RULES:
        if pattern.search(text):
            matches.append(
                {
                    "rule": rule_name,
                    "reason": explanation,
                }
            )

    return matches


def security_check(user_input: str) -> Dict[str, object]:
    """
    Check user input before it is passed to an LLM.

    PII is masked for allowed messages.
    Prompt-injection messages are blocked.
    """
    # Basic input check for empty or invalid values.
    if not isinstance(user_input, str) or not user_input.strip():
        return {
            "status": "blocked",
            "allowed": False,
            "reason": "Input must be a non-empty text message.",
            "sanitized_input": None,
            "pii_found": [],
            "injection_matches": [],
        }

    # First detect and mask PII.
    pii_result = mask_pii(user_input)

    # Then check the original message for prompt injection.
    injection_matches = detect_prompt_injection(user_input)

    # If an injection rule matches, block the message.
    if injection_matches:
        reasons = "; ".join(
            match["reason"] for match in injection_matches
        )

        return {
            "status": "blocked",
            "allowed": False,
            "reason": (
                f"Prompt-injection heuristic matched: {reasons}."
            ),
            "sanitized_input": None,
            "pii_found": pii_result["found_types"],
            "injection_matches": injection_matches,
        }

    # Otherwise, allow the sanitized input to continue.
    pii_found = pii_result["found_types"]

    if pii_found:
        reason = f"PII masked ({', '.join(pii_found)})."
    else:
        reason = "No supported PII pattern found."

    return {
        "status": "allowed",
        "allowed": True,
        "reason": reason,
        "sanitized_input": pii_result["text"],
        "pii_found": pii_found,
        "injection_matches": [],
    }


def handle_user_input(
    user_input: str,
    llm_call: Callable[[str], str],
) -> Dict[str, object]:
    """
    Enforce the security check before calling the LLM.

    The LLM receives sanitized input only.
    Blocked input never reaches the LLM.
    """
    check = security_check(user_input)

    # Important security gate.
    if not check["allowed"]:
        return check

    # Pass sanitized_input, not the original user_input, to the LLM.
    llm_response = llm_call(check["sanitized_input"])

    return {
        **check,
        "llm_response": llm_response,
    }


