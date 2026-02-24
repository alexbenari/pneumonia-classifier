PROMPT_TEMPLATES = {
    "v1_strict": (
        "You are given one chest X-ray image. "
        "Classify it into exactly one label from this set: normal, pneumonia. "
        "Output exactly one lowercase word: normal or pneumonia. "
        "Do not output any other text."
    )
}


def get_prompt_text(prompt_cfg):
    prompt_id = prompt_cfg.get("prompt_id", "v1_strict")
    configured_text = prompt_cfg.get("text", "")
    if configured_text and configured_text.strip():
        return prompt_id, configured_text.strip()
    if prompt_id in PROMPT_TEMPLATES:
        return prompt_id, PROMPT_TEMPLATES[prompt_id]
    raise ValueError(
        f"Unknown prompt_id '{prompt_id}' and no explicit prompt.text provided."
    )
