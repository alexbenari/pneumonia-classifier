import os

from ChestXrayDataset import VALID_IMAGE_EXTS


def normalize_class_name(class_name):
    name = class_name.strip().lower()
    if name == "normal":
        return "normal"
    if name == "pneumonia":
        return "pneumonia"
    return None


def load_split_records(data_dir, split):
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        raise ValueError(f"Split directory not found: {split_dir}")

    class_counts = {"normal": 0, "pneumonia": 0}
    records = []

    for class_name in sorted(os.listdir(split_dir)):
        class_path = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        normalized_label = normalize_class_name(class_name)
        if normalized_label is None:
            raise ValueError(
                f"Unsupported class folder '{class_name}' in split '{split}'. "
                "Expected class folders that normalize to 'normal' and 'pneumonia'."
            )

        for image_name in sorted(os.listdir(class_path)):
            image_path = os.path.join(class_path, image_name)
            if not os.path.isfile(image_path):
                continue
            _, ext = os.path.splitext(image_name)
            if ext.lower() not in VALID_IMAGE_EXTS:
                continue
            class_counts[normalized_label] += 1
            records.append(
                {
                    "image_path": image_path,
                    "true_label": normalized_label,
                }
            )

    if not records:
        raise ValueError(f"No image records found in split directory: {split_dir}")

    return records, class_counts
