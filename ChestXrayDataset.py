from torch.utils.data import Dataset
import os
from PIL import Image

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"}


class ChestXrayDataset(Dataset):
    def __init__(self, data_dir, transform=None, class_to_idx=None, strict=True):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self._valid_exts = VALID_IMAGE_EXTS
        if class_to_idx is None:
            self.class_to_idx = {
                class_name: i
                for i, class_name in enumerate(
                    sorted(
                        d for d in os.listdir(data_dir)
                        if os.path.isdir(os.path.join(data_dir, d))
                    )
                )
            }
        else:
            self.class_to_idx = dict(class_to_idx)
            if strict:
                actual_classes = sorted(
                    d for d in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir, d))
                )
                expected_classes = sorted(self.class_to_idx.keys())
                missing = [c for c in expected_classes if c not in actual_classes]
                extra = [c for c in actual_classes if c not in expected_classes]
                if missing or extra:
                    raise ValueError(
                        "Class mismatch in dataset split. "
                        f"Missing: {missing or 'none'}, Extra: {extra or 'none'}."
                    )

        for class_name, label in self.class_to_idx.items():
            class_path = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_path):
                if strict:
                    raise ValueError(f"Missing class folder '{class_name}' in {data_dir}.")
                continue
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if not os.path.isfile(img_path):
                    continue
                _, ext = os.path.splitext(img_name)
                if ext.lower() not in self._valid_exts:
                    continue
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
