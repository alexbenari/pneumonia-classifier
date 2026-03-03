import os


def get_prompt_text(prompt_cfg):
    prompt_id = prompt_cfg.get("prompt_id", "custom")
    prompt_path = (prompt_cfg.get("path", "") or "").strip()
    if not prompt_path:
        raise ValueError("prompt.path is required and must point to a prompt text file.")
    if not os.path.isfile(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()
    if not prompt_text:
        raise ValueError(f"Prompt file is empty: {prompt_path}")

    return prompt_id, prompt_text, prompt_path
