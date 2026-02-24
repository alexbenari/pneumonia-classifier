import re


def normalize_generated_text(raw_text):
    normalized = (raw_text or "").strip().lower()
    normalized = normalized.replace("\r", "\n").split("\n", 1)[0].strip()
    normalized = normalized.strip(" \t\"'`.,;:!?()[]{}")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def parse_label(raw_text, allowed_labels):
    normalized = normalize_generated_text(raw_text)
    allowed = {label.strip().lower() for label in allowed_labels}
    if normalized in allowed:
        return normalized, normalized
    return None, normalized
