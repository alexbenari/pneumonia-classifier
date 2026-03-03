# MedGemma Prompt Draft (Single-Call A/B)

## Draft v1

```text
You are given one chest X-ray image.

Task:
Classify the image into exactly one class.

Class definitions:
A = normal
Choose A if there is no clear radiographic evidence of pneumonia.

B = pneumonia
Choose B if the chest X-ray shows radiographic findings consistent with pneumonia.

Output format:
Output exactly one uppercase letter: A or B.
Do not output any other text.
```

## Label mapping

- `A` -> `normal`
- `B` -> `pneumonia`

## Notes to refine

- Keep output constrained to one token-like symbol (`A`/`B`) for single-call scoring.
- Adjust clinical wording to improve sensitivity for pneumonia without overcalling normal variants.
- If needed, add a short tie-break rule (for uncertain/borderline findings).
