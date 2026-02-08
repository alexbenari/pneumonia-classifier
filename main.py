import argparse
import json
import os
import re
from datetime import datetime
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
import warnings
import time

def main():
    warnings.filterwarnings('ignore')
    import torch
    torch.backends.cudnn.benchmark = True
    from torchvision import models
    import torch.optim as optim
    import torch.nn as nn
    from ChestXrayDataset import ChestXrayDataset
    from helper import count_images, report, write_classification_report, compare_training_strategies

    print("Cuda available: ", torch.cuda.is_available())
    print("Device count: ", torch.cuda.device_count())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
    print("torch.__version__:", torch.__version__)
    print(torch.version.cuda)

    base_dir = ".\\datasets"
    train_dir = os.path.join(base_dir, "train")
    val_dir   = os.path.join(base_dir, "val")
    test_dir  = os.path.join(base_dir, "test")

    parser = get_cli_arg_parser()
    args = parser.parse_args()

    cfg_name, cfg = get_run_hp_configuration(args)
    
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

    model_name = cfg["model_name"]
    match = re.fullmatch(r"resnet(\d+)", model_name)
    if not match:
        raise ValueError(f"Unsupported model_name '{model_name}'. Expected like 'resnet50'.")
    resnet_size = int(match.group(1))

    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    print("Classes:", class_names)

    output_dir = os.path.join("outputs", run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"=== Run: {run_name} ===")

    train_transform = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  # zoom + shift approximation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.10, scale=(0.01, 0.03), ratio=(0.3, 3.3), value=0),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    batch_size = cfg["batch_size"]
    num_workers = cfg["num_workers"]
    pin_memory = cfg.get("pin_memory", True)

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

    def freeze_bn(module):
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()

    def get_resnet(size):
        if size == 50:
            model = models.resnet50(pretrained=True)
        elif size == 101:
            model = models.resnet101(pretrained=True)
        else:
            raise ValueError("Unsupported ResNet size")
        return model


    def build_optimizer(model, lr, weight_decay_backbone, weight_decay_head, use_adamw):
        decay_backbone, no_decay_backbone = [], []
        decay_head, no_decay_head = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_head = name.startswith("fc.")
            is_no_decay = name.endswith(".bias") or "bn" in name.lower()
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

    def train_model(model, epochs,train_loader, validation_loader, optimizer, criterion):
        num_epochs = epochs
        train_losses = []
        val_losses   = []
        train_accs   = []
        val_accs     = []
        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

        for epoch in range(num_epochs):
            if epoch == 0:
                epoch_start = time.perf_counter()
            model.train()
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

    def eval_model(model, data_loader, criterion):
        model.eval()
        all_labels = []
        all_preds  = []
        running_loss, total = 0.0, 0
        use_amp = device.type == "cuda"
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
    resnet = get_resnet(resnet_size)
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)
    resnet = resnet.to(device)
    for param in resnet.parameters():
        param.requires_grad = False

    for param in resnet.fc.parameters():
        param.requires_grad = True
    #resnet.apply(freeze_bn)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = build_optimizer(
        resnet,
        cfg["lr_frozen"],
        weight_decay_backbone,
        weight_decay_head,
        use_adamw,
    )

    train_losses_frozen, train_accs_frozen, val_losses_frozen, val_accs_frozen = train_model(
        resnet, cfg["epochs_frozen"], train_loader, val_loader, optimizer, criterion)
    eval_labels_frozen, eval_preds_frozen, test_loss_frozen, test_acc_frozen = eval_model(
        resnet, test_loader, criterion)
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
    resnet = get_resnet(resnet_size)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    resnet = resnet.to(device)
    for param in resnet.parameters():
        param.requires_grad = True

    optimizer = build_optimizer(
        resnet,
        cfg["lr_full"],
        weight_decay_backbone,
        weight_decay_head,
        use_adamw,
    )

    train_losses_full, train_accs_full, val_losses_full, val_accs_full = train_model(
        resnet, cfg["epochs_full"], train_loader, val_loader, optimizer, criterion)
    eval_labels_full, eval_preds_full, test_loss_full, test_acc_full = eval_model(
        resnet, test_loader, criterion)
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


    #partial fine tuning - unfreeze last two layers
    print("Partial fine tuning - replace last layer and fully connected layer")
    resnet = get_resnet(resnet_size)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    resnet = resnet.to(device)

    for param in resnet.parameters():
        param.requires_grad = False

    for param in resnet.layer4.parameters():
        param.requires_grad = True

    for param in resnet.fc.parameters():
        param.requires_grad = True

    #resnet.apply(freeze_bn)

    optimizer = build_optimizer(
        resnet,
        cfg["lr_partial"],
        weight_decay_backbone,
        weight_decay_head,
        use_adamw,
    )

    train_losses_partial, train_accs_partial, val_losses_partial, val_accs_partial = train_model(
        resnet, cfg["epochs_partial"], train_loader, val_loader, optimizer, criterion)
    eval_labels_partial, eval_preds_partial, test_loss_partial, test_acc_partial = eval_model(
        resnet, test_loader, criterion)
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

    compare_training_strategies(cfg_name, cfg, optimizer_name, weight_decay_backbone, weight_decay_head, timestamp, run_tag, run_name, output_dir, batch_size, num_workers, report_dict_frozen, report_dict_full, report_dict_partial, strategies, test_accuracies, test_losses, final_train_loss, final_val_loss, final_train_acc, final_val_acc, overfit_gap)


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
