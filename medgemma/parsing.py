import re


def normalize_generated_text(raw_text):
    normalized = (raw_text or "").strip().lower()
    normalized = normalized.replace("\r", "\n").split("\n", 1)[0].strip()
    normalized = normalized.strip(" \t\"'`.,;:!?()[]{}")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def build_response_label_map(raw_map, allowed_labels):
    if not raw_map:
        return {}

    allowed = {label.strip().lower() for label in allowed_labels}
    response_map = {}
    for raw_response, raw_label in raw_map.items():
        response_key = normalize_generated_text(raw_response)
        label_value = normalize_generated_text(raw_label)
        if not response_key:
            continue
        if label_value not in allowed:
            raise ValueError(
                f"Invalid parsing.response_to_label target '{raw_label}'. "
                f"Allowed labels: {sorted(allowed)}"
            )
        response_map[response_key] = label_value
    return response_map


def parse_label(raw_text, allowed_labels, response_to_label=None):
    normalized = normalize_generated_text(raw_text)
    allowed = {label.strip().lower() for label in allowed_labels}
    if normalized in allowed:
        return normalized, normalized

    response_map = build_response_label_map(response_to_label, allowed_labels)
    if not response_map:
        return None, normalized

    if normalized in response_map:
        return response_map[normalized], normalized

    tokens = re.findall(r"[a-z]+", normalized)
    if tokens:
        first_token = tokens[0]
        if first_token in response_map:
            return response_map[first_token], normalized

        found_labels = []
        for token in tokens:
            mapped = response_map.get(token)
            if mapped is not None:
                found_labels.append(mapped)
        found_unique = sorted(set(found_labels))
        if len(found_unique) == 1:
            return found_unique[0], normalized

    return None, normalized
