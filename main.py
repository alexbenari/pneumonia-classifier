import argparse
import json
import os
import re
import random
from datetime import datetime
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
#import warnings
import time

def main():
    #warnings.filterwarnings('ignore')
    import torch
    torch.backends.cudnn.benchmark = True
    from torchvision import models
    import torch.optim as optim
    import torch.nn as nn
    from ChestXrayDataset import ChestXrayDataset
    from helper import count_images, report, write_classification_report, compare_training_strategies

    check_for_cuda(torch)

    base_dir = ".\\datasets"
    train_dir = os.path.join(base_dir, "train")
    val_dir   = os.path.join(base_dir, "val")
    test_dir  = os.path.join(base_dir, "test")

    parser = get_cli_arg_parser()
    args = parser.parse_args()

    cfg_name, cfg = get_run_hp_configuration(args)

    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    data_rng = torch.Generator()
    data_rng.manual_seed(seed)

    optimizer_name = cfg["optimizer"]
    use_adamw = optimizer_name.lower() == "adamw"
    if use_adamw:
        weight_decay_backbone = cfg["weight_decay_backbone"]
        weight_decay_head = cfg["weight_decay_head"]
    else:
        weight_decay_backbone = 0.0
        weight_decay_head = 0.0

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    raw_run_tag = args.run_tag.strip()
    run_tag = re.sub(r"[^A-Za-z0-9_-]+", "-", raw_run_tag).strip("-_") if raw_run_tag else ""
    run_prefix = f"{cfg_name}_{run_tag}" if run_tag else cfg_name
    run_name = (
        f"{run_prefix}_{cfg['model_name']}_{optimizer_name}_"
        f"wdback{weight_decay_backbone:g}_wdhead{weight_decay_head:g}_"
        f"{timestamp}"
    )

    ARCH_FINAL_HEAD = {
        "resnet": "fc",
        "densenet": "classifier",
        "efficientnet": "classifier.1",
        "vit": "heads.head",
    }
    if cfg.get("arch") not in ARCH_FINAL_HEAD:
        raise ValueError(f"Unsupported arch '{cfg.get('arch')}'.")

    model_name = cfg["model_name"]

    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    print("Classes:", class_names)

    output_dir = os.path.join("outputs", run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"=== Run: {run_name} ===")

    rotation_degrees = cfg.get("rotation_degrees", 20)
    enable_random_erasing = cfg.get("transform-random-erasing", False)
    train_transforms = [
        #transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(rotation_degrees),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  # zoom + shift approximation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
    if enable_random_erasing:
        train_transforms.append(
            transforms.RandomErasing(p=0.10, scale=(0.01, 0.03), ratio=(0.3, 3.3), value=0)
        )
    train_transform = transforms.Compose(train_transforms)

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    batch_size = cfg["batch_size"]
    num_workers = cfg["num_workers"]
    pin_memory = cfg.get("pin_memory", True)
    freeze_norm_layer = cfg.get("freeze_norm_layer", False)

    def seed_worker(worker_id):
        worker_seed = (seed + worker_id) % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    train_dataset = ChestXrayDataset(train_dir, transform=train_transform, class_to_idx=class_to_idx, strict=True)
    val_dataset   = ChestXrayDataset(val_dir,   transform=val_test_transform, class_to_idx=class_to_idx, strict=True)
    test_dataset  = ChestXrayDataset(test_dir,  transform=val_test_transform, class_to_idx=class_to_idx, strict=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=data_rng,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    train_counts, train_total = count_images(train_dir, class_names)
    val_counts, val_total     = count_images(val_dir, class_names)
    test_counts, test_total   = count_images(test_dir, class_names)
    print("Train samples:", train_total, train_counts)
    print("Validation samples:", val_total, val_counts)
    print("Test samples:", test_total, test_counts)

    num_classes = len(class_names)
    train_labels = [label for _, label in train_dataset.samples]

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print("Class weights:", class_weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    def freeze_norms(module):
        if isinstance(
            module,
            (
                torch.nn.modules.batchnorm._BatchNorm,
                torch.nn.InstanceNorm1d,
                torch.nn.InstanceNorm2d,
                torch.nn.InstanceNorm3d,
                torch.nn.LayerNorm,
                torch.nn.GroupNorm,
            ),
        ):
            module.eval()

    def get_model(arch, model_name):
        if arch == "resnet":
            if model_name == "resnet50":
                return models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            if model_name == "resnet101":
                return models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            raise ValueError(f"Unsupported ResNet model_name '{model_name}'")
        if arch == "densenet":
            if model_name == "densenet121":
                return models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            if model_name == "densenet169":
                return models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
            if model_name == "densenet201":
                return models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
            raise ValueError(f"Unsupported DenseNet model_name '{model_name}'")
        if arch == "efficientnet":
            if model_name in {
                "efficientnet_b0",
                "efficientnet_b1",
                "efficientnet_b2",
                "efficientnet_b3",
                "efficientnet_b4",
                "efficientnet_b5",
                "efficientnet_b6",
                "efficientnet_b7",
            }:
                weights_enum = getattr(models, f"{model_name.upper()}_Weights")
                return getattr(models, model_name)(weights=weights_enum.DEFAULT)
            raise ValueError(f"Unsupported EfficientNet model_name '{model_name}'")
        if arch == "vit":
            if model_name in {"vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14"}:
                weights_enum = getattr(models, f"{model_name.upper()}_Weights")
                return getattr(models, model_name)(weights=weights_enum.DEFAULT)
            raise ValueError(f"Unsupported ViT model_name '{model_name}'")
        raise ValueError(f"Unsupported arch '{arch}'")

    def replace_head(model, arch, num_classes):
        if arch == "resnet":
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
        if arch == "densenet":
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            return model
        if arch == "efficientnet":
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            return model
        if arch == "vit":
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
            return model
        raise ValueError(f"Unsupported arch '{arch}'")

    def set_trainable_layers(model, arch, strategy):
        if strategy == "full":
            for param in model.parameters():
                param.requires_grad = True
            return

        for param in model.parameters():
            param.requires_grad = False

        head_module = model.get_submodule(ARCH_FINAL_HEAD[arch])
        for param in head_module.parameters(recurse=False):
            param.requires_grad = True

        if strategy == "frozen":
            return

        #partial fine-tuning - unfreeze last block + classifier
        if arch == "resnet":
            for param in model.layer4.parameters():
                param.requires_grad = True
            return
        if arch == "densenet":
            for param in model.features.denseblock4.parameters():
                param.requires_grad = True
            for param in model.features.norm5.parameters():
                param.requires_grad = True
            return
        if arch == "efficientnet":
            for param in model.features[-1].parameters():
                param.requires_grad = True
            return
        if arch == "vit":
            for param in model.encoder.layers[-1].parameters():
                param.requires_grad = True
            return

        raise ValueError(f"Partial fine-tuning not supported for arch '{arch}'")


    def build_optimizer(model, lr, weight_decay_backbone, weight_decay_head, use_adamw, head_module):
        decay_backbone, no_decay_backbone = [], []
        decay_head, no_decay_head = [], []
        no_decay_param_ids = set()

        # BatchNorm params should not have weight decay
        for module in model.modules():
            if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d, torch.nn.BatchNorm3d)):
                for param in module.parameters(recurse=False):
                    no_decay_param_ids.add(id(param))

        head = model.get_submodule(head_module)
        head_param_ids = {id(param) for param in head.parameters(recurse=False)}
        if not head_param_ids:
            raise ValueError(f"No head params found for head_module='{head_module}'")

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_head = id(param) in head_param_ids
            is_no_decay = name.endswith(".bias") or (id(param) in no_decay_param_ids)
            if is_head:
                if is_no_decay:
                    no_decay_head.append(param)
                else:
                    decay_head.append(param)
            else:
                if is_no_decay:
                    no_decay_backbone.append(param)
                else:
                    decay_backbone.append(param)

        if use_adamw:
            param_groups = []
            if decay_backbone:
                param_groups.append({"params": decay_backbone, "weight_decay": weight_decay_backbone})
            if no_decay_backbone:
                param_groups.append({"params": no_decay_backbone, "weight_decay": 0.0})
            if decay_head:
                param_groups.append({"params": decay_head, "weight_decay": weight_decay_head})
            if no_decay_head:
                param_groups.append({"params": no_decay_head, "weight_decay": 0.0})
            return optim.AdamW(param_groups, lr=lr)

        param_groups = []
        if decay_backbone:
            param_groups.append({"params": decay_backbone, "weight_decay": weight_decay_backbone})
        if no_decay_backbone:
            param_groups.append({"params": no_decay_backbone, "weight_decay": 0.0})
        if decay_head:
            param_groups.append({"params": decay_head, "weight_decay": weight_decay_head})
        if no_decay_head:
            param_groups.append({"params": no_decay_head, "weight_decay": 0.0})
        return optim.Adam(param_groups, lr=lr)

    def train_model(model, epochs,train_loader, validation_loader, optimizer, criterion, use_amp, freeze_norm_layer):
        num_epochs = epochs
        train_losses = []
        val_losses   = []
        train_accs   = []
        val_accs     = []
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

        for epoch in range(num_epochs):
            if epoch == 0:
                epoch_start = time.perf_counter()
            model.train()
            if freeze_norm_layer:
                model.apply(freeze_norms)
            running_loss, correct, total = 0.0, 0, 0

            for images, labels in train_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_train_loss = running_loss / total
            epoch_train_acc  = correct / total
            train_losses.append(epoch_train_loss)
            train_accs.append(epoch_train_acc)

            model.eval()
            running_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in validation_loader:
                    images, labels = images.to(device), labels.to(device)
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    running_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            epoch_val_loss = running_loss / total
            epoch_val_acc  = correct / total
            val_losses.append(epoch_val_loss)
            val_accs.append(epoch_val_acc)
            if epoch == 0:
                epoch_seconds = time.perf_counter() - epoch_start
                print(f"time per epoch: {epoch_seconds:.2f}s")
            print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

        return train_losses, train_accs, val_losses, val_accs

    def eval_model(model, data_loader, criterion, use_amp):
        model.eval()
        all_labels = []
        all_preds  = []
        running_loss, total = 0.0, 0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                running_loss += loss.item() * images.size(0)
                total += labels.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        test_acc = sum(np.array(all_labels) == np.array(all_preds)) / len(all_labels)
        test_loss = running_loss / total if total > 0 else float("nan")
        return all_labels, all_preds, test_loss, test_acc

    def save_eval(output_dir, strategy_id, labels, preds, class_names):
        path = os.path.join(output_dir, f"eval_{strategy_id}.npz")
        np.savez(
            path,
            labels=np.array(labels),
            preds=np.array(preds),
            class_names=np.array(class_names),
            strategy=strategy_id,
        )

    print("FC Only Training - replace only final FC layer")
    model = get_model(cfg["arch"], model_name)
    model = replace_head(model, cfg["arch"], num_classes)
    model = model.to(device)
    set_trainable_layers(model, cfg["arch"], "frozen")
    #model.apply(freeze_bn)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = build_optimizer(
        model,
        cfg["lr_frozen"],
        weight_decay_backbone,
        weight_decay_head,
        use_adamw,
        ARCH_FINAL_HEAD[cfg["arch"]],
    )

    use_amp = device.type == "cuda"

    train_losses_frozen, train_accs_frozen, val_losses_frozen, val_accs_frozen = train_model(
        model, cfg["epochs_frozen"], train_loader, val_loader, optimizer, criterion, use_amp, freeze_norm_layer)
    eval_labels_frozen, eval_preds_frozen, test_loss_frozen, test_acc_frozen = eval_model(
        model, test_loader, criterion, use_amp)
    print(f"Test Accuracy (FC Only): {test_acc_frozen*100:.2f}% | Test Loss: {test_loss_frozen:.4f}")
    save_eval(output_dir, "fc_only", eval_labels_frozen, eval_preds_frozen, class_names)
    report_dict_frozen, report_text_frozen = write_classification_report(
        output_dir, "fc_only", eval_labels_frozen, eval_preds_frozen, class_names
    )

    report(
        "FC Only",
        train_losses_frozen,
        val_losses_frozen, 
        train_accs_frozen,
        val_accs_frozen,
        eval_labels_frozen,
        eval_preds_frozen,
        class_names,
        output_dir=output_dir,
        classification_report_text=report_text_frozen,
    )


    print("Full network fine tuning")
    model = get_model(cfg["arch"], model_name)
    model = replace_head(model, cfg["arch"], num_classes)
    model = model.to(device)
    set_trainable_layers(model, cfg["arch"], "full")

    optimizer = build_optimizer(
        model,
        cfg["lr_full"],
        weight_decay_backbone,
        weight_decay_head,
        use_adamw,
        ARCH_FINAL_HEAD[cfg["arch"]],
    )

    train_losses_full, train_accs_full, val_losses_full, val_accs_full = train_model(
        model, cfg["epochs_full"], train_loader, val_loader, optimizer, criterion, use_amp, freeze_norm_layer)
    eval_labels_full, eval_preds_full, test_loss_full, test_acc_full = eval_model(
        model, test_loader, criterion, use_amp)
    print(f"Test Accuracy (Full fine tuning): {test_acc_full*100:.2f}% | Test Loss: {test_loss_full:.4f}")
    save_eval(output_dir, "full_finetuning", eval_labels_full, eval_preds_full, class_names)
    report_dict_full, report_text_full = write_classification_report(
        output_dir, "full_finetuning", eval_labels_full, eval_preds_full, class_names
    )
    report(
        "Full Fine-Tuning",
        train_losses_full,
        val_losses_full, 
        train_accs_full,
        val_accs_full,
        eval_labels_full,
        eval_preds_full,
        class_names,
        output_dir=output_dir,
        classification_report_text=report_text_full,
    )


    #partial fine tuning - unfreeze last block + classifier
    print("Partial fine tuning - replace last layer and fully connected layer")
    model = get_model(cfg["arch"], model_name)
    model = replace_head(model, cfg["arch"], num_classes)
    model = model.to(device)
    set_trainable_layers(model, cfg["arch"], "partial")
    #model.apply(freeze_bn)

    optimizer = build_optimizer(
        model,
        cfg["lr_partial"],
        weight_decay_backbone,
        weight_decay_head,
        use_adamw,
        ARCH_FINAL_HEAD[cfg["arch"]],
    )

    train_losses_partial, train_accs_partial, val_losses_partial, val_accs_partial = train_model(
        model, cfg["epochs_partial"], train_loader, val_loader, optimizer, criterion, use_amp, freeze_norm_layer)
    eval_labels_partial, eval_preds_partial, test_loss_partial, test_acc_partial = eval_model(
        model, test_loader, criterion, use_amp)
    print(f"Test Accuracy (Partial fine tuning): {test_acc_partial*100:.2f}% | Test Loss: {test_loss_partial:.4f}")
    save_eval(output_dir, "partial_finetuning", eval_labels_partial, eval_preds_partial, class_names)
    report_dict_partial, report_text_partial = write_classification_report(
        output_dir, "partial_finetuning", eval_labels_partial, eval_preds_partial, class_names
    )
    report(
        "Partial Fine-Tuning",
        train_losses_partial,
        val_losses_partial, 
        train_accs_partial,
        val_accs_partial,
        eval_labels_partial,
        eval_preds_partial,
        class_names,
        output_dir=output_dir,
        classification_report_text=report_text_partial,
    )

    #comparative analysis 
    strategies = ['FC Only', 'Full Fine-Tuning', 'Partial Fine-Tuning']
    test_accuracies = [test_acc_frozen, test_acc_full, test_acc_partial]
    test_losses = [test_loss_frozen, test_loss_full, test_loss_partial]

    final_train_loss = [train_losses_frozen[-1], train_losses_full[-1], train_losses_partial[-1]]
    final_val_loss   = [val_losses_frozen[-1], val_losses_full[-1], val_losses_partial[-1]]

    final_train_acc = [train_accs_frozen[-1], train_accs_full[-1], train_accs_partial[-1]]
    final_val_acc   = [val_accs_frozen[-1], val_accs_full[-1], val_accs_partial[-1]]
    overfit_gap = [t - v for t, v in zip(final_train_acc, final_val_acc)]

    compare_training_strategies(
        cfg_name,
        cfg,
        optimizer_name,
        ARCH_FINAL_HEAD[cfg["arch"]],
        weight_decay_backbone,
        weight_decay_head,
        timestamp,
        run_tag,
        run_name,
        output_dir,
        batch_size,
        num_workers,
        report_dict_frozen,
        report_dict_full,
        report_dict_partial,
        strategies,
        test_accuracies,
        test_losses,
        final_train_loss,
        final_val_loss,
        final_train_acc,
        final_val_acc,
        overfit_gap,
    )

def check_for_cuda(torch):
    print("Cuda available: ", torch.cuda.is_available())
    print("Device count: ", torch.cuda.device_count())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
    print("torch.__version__:", torch.__version__)
    print(torch.version.cuda)


def get_training_run_name(args):
    raw_run_tag = args.run_tag.strip()
    run_tag = re.sub(r"[^A-Za-z0-9_-]+", "-", raw_run_tag).strip("-_") if raw_run_tag else ""
    return run_tag

def get_run_hp_configuration(args):
    config_path = os.path.join(os.path.dirname(__file__), "configs.json")
    with open(config_path, "r", encoding="utf-8") as f:
        all_cfgs = json.load(f)
    if args.cfg not in all_cfgs:
        raise ValueError(f"Config '{args.cfg}' not found in {config_path}")
    cfg_name = args.cfg
    cfg = all_cfgs[cfg_name]
    return cfg_name,cfg

def get_cli_arg_parser():
    parser = argparse.ArgumentParser(description="Train pneumonia classifier")
    parser.add_argument("--cfg", default="default", help="Config name from configs.json")
    parser.add_argument("--run-tag", default="", help="Optional run tag for run name prefix")
    return parser


if __name__ == '__main__':
    main()
