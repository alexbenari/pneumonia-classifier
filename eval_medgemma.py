import argparse
import json
import os
import random
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import AutoProcessor, BitsAndBytesConfig
from huggingface_hub.errors import GatedRepoError

from medgemma.dataset import load_split_records
from medgemma.parsing import parse_label
from medgemma.prompts import get_prompt_text


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot MedGemma evaluation.")
    parser.add_argument(
        "--split",
        required=True,
        choices=["val", "test"],
        help="Dataset split to evaluate (val or test).",
    )
    parser.add_argument(
        "--model-id",
        default="google/medgemma-1.5-4b-it",
        help="Model id used for both loading and config-file lookup.",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help="Optional run tag added to output run directory name.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of samples for quick tests.",
    )
    return parser.parse_args()


def model_id_to_config_path(model_id):
    model_slug = model_id.replace("/", "__")
    return os.path.join("models", f"{model_slug}.json")


def sanitize_tag(raw_tag):
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", (raw_tag or "").strip())
    return cleaned.strip("-_")


def load_model_config(model_id):
    config_path = model_id_to_config_path(model_id)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Model config not found: {config_path}. "
            "Create it before running evaluation."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config_path, config


def resolve_dtype(dtype_name):
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(
        f"Unsupported dtype '{dtype_name}'. Use one of: bfloat16, float16, float32."
    )


def choose_runtime_device(device_preference):
    if device_preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested CUDA but no CUDA device is available.")
    if device_preference not in {"cuda", "cpu"}:
        raise ValueError("model.device must be one of: auto, cuda, cpu.")
    return device_preference


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_quantization_config(model_cfg):
    load_in_4bit = bool(model_cfg.get("load_in_4bit", False))
    load_in_8bit = bool(model_cfg.get("load_in_8bit", False))
    cpu_offload = bool(model_cfg.get("llm_int8_enable_fp32_cpu_offload", False))
    if load_in_4bit and load_in_8bit:
        raise ValueError("Only one of load_in_4bit/load_in_8bit can be true.")
    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_enable_fp32_cpu_offload=cpu_offload,
            bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=resolve_dtype(
                model_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
            ),
        )
    if load_in_8bit:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=cpu_offload,
        )
    return None


def load_model_and_processor(model_cfg):
    try:
        from transformers import AutoModelForImageTextToText

        model_class = AutoModelForImageTextToText
    except ImportError:
        try:
            from transformers import Gemma3ForConditionalGeneration

            model_class = Gemma3ForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "No MedGemma-compatible model class found in transformers. "
                "Upgrade transformers to a recent version that supports MedGemma/Gemma3."
            ) from exc

    model_id = model_cfg["model_id"]
    cache_dir = model_cfg.get("hf_cache_dir")
    runtime_device = choose_runtime_device(model_cfg.get("device", "auto"))

    configured_dtype_name = model_cfg.get("dtype", "bfloat16")
    if runtime_device == "cpu" and configured_dtype_name in {"bfloat16", "float16"}:
        dtype_name = "float32"
    else:
        dtype_name = configured_dtype_name
    torch_dtype = resolve_dtype(dtype_name)

    quantization_config = build_quantization_config(model_cfg)
    model_kwargs = {
        "cache_dir": cache_dir,
        "dtype": torch_dtype,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = model_cfg.get("device_map", "auto")
    else:
        model_kwargs["device_map"] = "auto" if runtime_device == "cuda" else "cpu"

    offload_folder = model_cfg.get("offload_folder")
    if offload_folder:
        os.makedirs(offload_folder, exist_ok=True)
        model_kwargs["offload_folder"] = offload_folder

    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if hf_token:
        model_kwargs["token"] = hf_token
        processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=cache_dir, token=hf_token
        )
    else:
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)

    model = model_class.from_pretrained(model_id, **model_kwargs)
    model.eval()

    return processor, model, runtime_device, dtype_name


def get_model_device(model):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def build_messages(prompt_text, image):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def prepare_inputs(processor, prompt_text, image):
    messages = build_messages(prompt_text, image)
    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except TypeError:
        chat_text = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(text=chat_text, images=image, return_tensors="pt")

    if "input_ids" not in inputs:
        raise RuntimeError("Processor output missing input_ids.")
    prompt_length = int(inputs["input_ids"].shape[-1])
    return inputs, prompt_length


def build_label_token_map(processor, allowed_labels, label_prefix=""):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Processor does not expose a tokenizer for label scoring.")

    token_map = {}
    for label in allowed_labels:
        label_text = f"{label_prefix}{label}"
        encoded = tokenizer(label_text, add_special_tokens=False, return_tensors="pt")
        candidate_ids = encoded.get("input_ids")
        if candidate_ids is None or candidate_ids.shape[-1] == 0:
            raise RuntimeError(f"Failed to tokenize label candidate '{label_text}'.")
        token_map[label] = candidate_ids
    return token_map


def score_candidate_label(model, prepared_inputs, prompt_length, candidate_ids, score_reduction):
    candidate_len = int(candidate_ids.shape[-1])
    full_inputs = {}

    for key, value in prepared_inputs.items():
        if not torch.is_tensor(value):
            full_inputs[key] = value
            continue

        if value.ndim == 2 and value.shape[-1] == prompt_length:
            if key == "input_ids":
                full_inputs[key] = torch.cat([value, candidate_ids], dim=-1)
            elif key == "attention_mask":
                extension = torch.ones(
                    (value.shape[0], candidate_len), dtype=value.dtype, device=value.device
                )
                full_inputs[key] = torch.cat([value, extension], dim=-1)
            elif key == "token_type_ids":
                extension = value[:, -1:].expand(-1, candidate_len)
                full_inputs[key] = torch.cat([value, extension], dim=-1)
            else:
                extension = torch.zeros(
                    (value.shape[0], candidate_len), dtype=value.dtype, device=value.device
                )
                full_inputs[key] = torch.cat([value, extension], dim=-1)
        else:
            full_inputs[key] = value

    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_length] = -100
    full_inputs["labels"] = labels

    with torch.inference_mode():
        outputs = model(**full_inputs)

    mean_logprob = -float(outputs.loss.item())
    if score_reduction == "sum_logprob":
        return mean_logprob * candidate_len
    if score_reduction == "mean_logprob":
        return mean_logprob
    raise ValueError("inference.score_reduction must be one of: mean_logprob, sum_logprob.")


def classify_by_label_scoring(
    model,
    processor,
    prompt_text,
    image,
    allowed_labels,
    label_token_map,
    score_reduction,
):
    inputs, prompt_length = prepare_inputs(processor, prompt_text, image)
    target_device = get_model_device(model)
    prepared_inputs = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            prepared_inputs[key] = value.to(target_device)
        else:
            prepared_inputs[key] = value

    scores = {}
    for label in allowed_labels:
        candidate_ids = label_token_map[label].to(target_device)
        scores[label] = score_candidate_label(
            model=model,
            prepared_inputs=prepared_inputs,
            prompt_length=prompt_length,
            candidate_ids=candidate_ids,
            score_reduction=score_reduction,
        )

    best_label = max(allowed_labels, key=lambda label: scores[label])
    return best_label, scores


def decode_label(model, processor, prompt_text, image, decoding_cfg):
    inputs, prompt_length = prepare_inputs(processor, prompt_text, image)
    target_device = get_model_device(model)
    for key, value in inputs.items():
        if torch.is_tensor(value):
            inputs[key] = value.to(target_device)

    do_sample = bool(decoding_cfg.get("do_sample", False))
    generate_kwargs = {
        "do_sample": do_sample,
        "num_beams": int(decoding_cfg.get("num_beams", 1)),
        "max_new_tokens": int(decoding_cfg.get("max_new_tokens", 4)),
    }
    if do_sample:
        generate_kwargs["temperature"] = float(decoding_cfg.get("temperature", 1.0))
        generate_kwargs["top_p"] = float(decoding_cfg.get("top_p", 1.0))

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and tokenizer.pad_token_id is not None:
        generate_kwargs["pad_token_id"] = int(tokenizer.pad_token_id)
    elif tokenizer is not None and tokenizer.eos_token_id is not None:
        generate_kwargs["pad_token_id"] = int(tokenizer.eos_token_id)

    with torch.inference_mode():
        generated = model.generate(**inputs, **generate_kwargs)

    generated_tokens = generated[:, prompt_length:]
    if generated_tokens.numel() == 0:
        return ""
    return processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]


def save_confusion_outputs(y_true, y_pred, labels, png_path, pdf_path, title):
    fig = plt.figure(figsize=(6, 5))
    if y_true:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=labels,
            yticklabels=labels,
            cmap="Blues",
        )
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.title(title)
    else:
        plt.axis("off")
        plt.text(0.5, 0.5, "No parseable predictions", ha="center", va="center")

    fig.savefig(png_path, bbox_inches="tight")
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def build_run_name(model_id, prompt_id, split, run_tag):
    model_slug = model_id.replace("/", "__")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    pieces = [model_slug, prompt_id, split]
    if run_tag:
        pieces.append(run_tag)
    pieces.append(timestamp)
    return "_".join(pieces), timestamp


def evaluate(args):
    config_path, config = load_model_config(args.model_id)
    model_cfg = config["model"]
    prompt_cfg = config["prompt"]
    decoding_cfg = config["decoding"]
    inference_cfg = config.get("inference", {})
    parsing_cfg = config["parsing"]
    io_cfg = config["io"]
    seed = int(config.get("seed", 42))

    prompt_id, prompt_text = get_prompt_text(prompt_cfg)
    allowed_labels = parsing_cfg.get("allowed_labels", ["normal", "pneumonia"])
    allowed_labels = [label.strip().lower() for label in allowed_labels]
    unparseable_policy = parsing_cfg.get("unparseable_policy", "exclude")
    if unparseable_policy != "exclude":
        raise ValueError(
            "Only unparseable_policy='exclude' is supported in this baseline script."
        )
    inference_mode = inference_cfg.get("mode", "generation")
    if inference_mode not in {"generation", "label_scoring"}:
        raise ValueError("inference.mode must be one of: generation, label_scoring.")
    score_reduction = inference_cfg.get("score_reduction", "mean_logprob")
    label_prefix = inference_cfg.get("label_prefix", "")

    set_seed(seed)

    try:
        processor, model, runtime_device, loaded_dtype = load_model_and_processor(model_cfg)
    except GatedRepoError as exc:
        raise RuntimeError(
            "Model download failed because this is a gated repo. "
            "Request access on Hugging Face and authenticate locally with `hf auth login` "
            "or set HF_TOKEN in the environment before running."
        ) from exc
    except OSError as exc:
        message = str(exc)
        if "gated repo" in message.lower():
            raise RuntimeError(
                "Model download failed due to gated access. "
                "Authenticate with Hugging Face (`hf auth login`) and ensure access "
                "to google/medgemma-1.5-4b-it is approved."
            ) from exc
        if "proxy" in message.lower():
            raise RuntimeError(
                "Model download failed due to proxy settings. "
                "Clear HTTP_PROXY/HTTPS_PROXY/ALL_PROXY for this run."
            ) from exc
        raise
    print(f"Loaded model: {model_cfg['model_id']}")
    print(f"Runtime device: {runtime_device} | dtype: {loaded_dtype}")

    records, class_counts = load_split_records(io_cfg.get("data_dir", "datasets"), args.split)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    run_tag = sanitize_tag(args.run_tag)
    run_name, timestamp = build_run_name(args.model_id, prompt_id, args.split, run_tag)
    output_root = io_cfg.get("output_root", "outputs-medgemma")
    output_dir = os.path.join(output_root, run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Samples to evaluate: {len(records)}")
    print(f"Inference mode: {inference_mode}")

    label_token_map = None
    if inference_mode == "label_scoring":
        label_token_map = build_label_token_map(
            processor=processor,
            allowed_labels=allowed_labels,
            label_prefix=label_prefix,
        )

    prediction_rows = []
    unparseable_rows = []
    y_true_eval = []
    y_pred_eval = []

    for idx, record in enumerate(records, start=1):
        image_path = record["image_path"]
        true_label = record["true_label"]
        with Image.open(image_path) as raw_image:
            image = raw_image.convert("RGB")
            if inference_mode == "label_scoring":
                raw_output, score_map = classify_by_label_scoring(
                    model=model,
                    processor=processor,
                    prompt_text=prompt_text,
                    image=image,
                    allowed_labels=allowed_labels,
                    label_token_map=label_token_map,
                    score_reduction=score_reduction,
                )
            else:
                raw_output = decode_label(
                    model=model,
                    processor=processor,
                    prompt_text=prompt_text,
                    image=image,
                    decoding_cfg=decoding_cfg,
                )
                score_map = {}

        parsed_label, _ = parse_label(raw_output, allowed_labels)
        is_parseable = parsed_label is not None

        row = {
            "image_path": image_path,
            "true_label": true_label,
            "raw_output": raw_output,
            "parsed_label": parsed_label,
            "is_parseable": is_parseable,
        }
        for label in allowed_labels:
            row[f"score_{label}"] = score_map.get(label)
        prediction_rows.append(row)

        if is_parseable:
            y_true_eval.append(true_label)
            y_pred_eval.append(parsed_label)
        else:
            unparseable_rows.append(
                {
                    "image_path": image_path,
                    "true_label": true_label,
                    "raw_output": raw_output,
                }
            )

        if idx % 25 == 0 or idx == len(records):
            print(f"Processed {idx}/{len(records)}")

    parseable_count = len(y_true_eval)
    total_count = len(prediction_rows)
    unparseable_count = len(unparseable_rows)
    parseable_rate = (parseable_count / total_count) if total_count else 0.0

    if parseable_count:
        report_dict = classification_report(
            y_true_eval,
            y_pred_eval,
            labels=allowed_labels,
            output_dict=True,
            zero_division=0,
        )
        accuracy = float(accuracy_score(y_true_eval, y_pred_eval))
        macro_f1 = float(f1_score(y_true_eval, y_pred_eval, average="macro", zero_division=0))
        weighted_f1 = float(
            f1_score(y_true_eval, y_pred_eval, average="weighted", zero_division=0)
        )
    else:
        report_dict = {}
        accuracy = None
        macro_f1 = None
        weighted_f1 = None

    per_class = {}
    for label in allowed_labels:
        label_metrics = report_dict.get(label, {})
        per_class[label] = {
            "precision": label_metrics.get("precision"),
            "recall": label_metrics.get("recall"),
            "f1": label_metrics.get("f1-score"),
            "support": int(label_metrics.get("support", 0)) if label_metrics else 0,
        }

    split = args.split
    predictions_csv = os.path.join(output_dir, f"predictions_{split}.csv")
    unparseable_csv = os.path.join(output_dir, f"unparseable_{split}.csv")
    report_json = os.path.join(output_dir, f"classification_report_{split}.json")
    summary_json = os.path.join(output_dir, f"summary_{split}.json")
    confusion_png = os.path.join(output_dir, f"confusion_matrix_{split}.png")
    confusion_pdf = os.path.join(output_dir, f"confusion_matrix_{split}.pdf")

    pd.DataFrame(prediction_rows).to_csv(predictions_csv, index=False)
    pd.DataFrame(
        unparseable_rows, columns=["image_path", "true_label", "raw_output"]
    ).to_csv(unparseable_csv, index=False)
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(json_safe(report_dict), f, indent=2)

    save_confusion_outputs(
        y_true=y_true_eval,
        y_pred=y_pred_eval,
        labels=allowed_labels,
        png_path=confusion_png,
        pdf_path=confusion_pdf,
        title=f"Confusion Matrix ({split}, parseable only)",
    )

    summary = {
        "run": {
            "name": run_name,
            "timestamp": timestamp,
            "split": split,
            "run_tag": run_tag or None,
        },
        "model": {
            "model_id": model_cfg["model_id"],
            "device": runtime_device,
            "dtype": loaded_dtype,
            "load_in_4bit": bool(model_cfg.get("load_in_4bit", False)),
            "load_in_8bit": bool(model_cfg.get("load_in_8bit", False)),
            "llm_int8_enable_fp32_cpu_offload": bool(
                model_cfg.get("llm_int8_enable_fp32_cpu_offload", False)
            ),
            "device_map": model_cfg.get("device_map", "auto"),
            "offload_folder": model_cfg.get("offload_folder"),
            "hf_cache_dir": model_cfg.get("hf_cache_dir"),
            "config_file": config_path,
        },
        "prompt": {
            "prompt_id": prompt_id,
            "text": prompt_text,
        },
        "decoding": {
            "do_sample": bool(decoding_cfg.get("do_sample", False)),
            "num_beams": int(decoding_cfg.get("num_beams", 1)),
            "temperature": float(decoding_cfg.get("temperature", 0.0)),
            "top_p": float(decoding_cfg.get("top_p", 1.0)),
            "max_new_tokens": int(decoding_cfg.get("max_new_tokens", 4)),
        },
        "inference": {
            "mode": inference_mode,
            "score_reduction": score_reduction,
            "label_prefix": label_prefix,
        },
        "dataset": {
            "data_dir": io_cfg.get("data_dir", "datasets"),
            "total_samples": total_count,
            "class_counts": class_counts,
            "max_samples": args.max_samples,
        },
        "parsing": {
            "allowed_labels": allowed_labels,
            "parseable_count": parseable_count,
            "unparseable_count": unparseable_count,
            "parseable_rate": parseable_rate,
            "unparseable_policy": unparseable_policy,
        },
        "metrics": {
            "evaluated_count": parseable_count,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "per_class": per_class,
        },
        "artifacts": {
            "predictions_csv": predictions_csv,
            "unparseable_csv": unparseable_csv,
            "classification_report_json": report_json,
            "confusion_png": confusion_png,
            "confusion_pdf": confusion_pdf,
        },
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2)

    print("=== Evaluation Complete ===")
    print(f"Parseable: {parseable_count}/{total_count} ({parseable_rate:.2%})")
    if accuracy is not None:
        print(f"Accuracy (parseable only): {accuracy:.4f}")
        print(f"Macro F1 (parseable only): {macro_f1:.4f}")
    else:
        print("No parseable predictions; metrics unavailable.")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    evaluate(parse_args())
