import argparse
import copy
import datetime as dt
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


LABEL_COLS = [
    "AKIEC",
    "BCC",
    "BEN_OTH",
    "BKL",
    "DF",
    "INF",
    "MAL_OTH",
    "MEL",
    "NV",
    "SCCKA",
    "VASC",
]

MONET_COLS = [
    "MONET_ulceration_crust",
    "MONET_hair",
    "MONET_vasculature_vessels",
    "MONET_erythema",
    "MONET_pigmented",
    "MONET_gel_water_drop_fluid_dermoscopy_liquid",
    "MONET_skin_markings_pen_ink_purple_pen",
]

MALIGNANT_CLASSES = {"AKIEC", "BCC", "MAL_OTH", "MEL", "SCCKA"}
ARTIFACT_MONET_COLS = {
    "MONET_hair",
    "MONET_gel_water_drop_fluid_dermoscopy_liquid",
    "MONET_skin_markings_pen_ink_purple_pen",
}


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"yes", "true", "t", "1", "y"}:
        return True
    if value in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def load_yaml_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    import yaml

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def parse_args() -> argparse.Namespace:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config_path", type=str, default=None)
    base_args, _ = base_parser.parse_known_args()
    cfg = load_yaml_config(base_args.config_path)

    parser = argparse.ArgumentParser(
        description="Run MILK10K black-box, predefined-concept, and discovered-concept experiments."
    )
    parser.add_argument("--config_path", type=str, default=base_args.config_path)
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default=cfg.get("metadata_csv", "data/MILK10K/MILK10k_Training_Metadata.csv"),
    )
    parser.add_argument(
        "--groundtruth_csv",
        type=str,
        default=cfg.get("groundtruth_csv", "data/MILK10K/MILK10k_Training_GroundTruth.csv"),
    )
    parser.add_argument(
        "--supplement_csv",
        type=str,
        default=cfg.get("supplement_csv", "data/MILK10K/MILK10k_Training_Supplement.csv"),
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=cfg.get("image_dir", "data/MILK10K/MILK10k_Training_Input/MILK10k_Training_Input"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=cfg.get("output_dir", "output/milk10k_concept_discovery"),
    )
    parser.add_argument("--seed", type=int, default=cfg.get("seed", 42))
    parser.add_argument("--limit", type=int, default=cfg.get("limit", 0))
    parser.add_argument(
        "--models",
        type=str,
        default=cfg.get("models", "E1,E2,E3,E4,E5,E6,E7"),
        help="Comma-separated experiment ids to run.",
    )

    parser.add_argument("--dinov2_model", type=str, default=cfg.get("dinov2_model", "vit_small_patch14_dinov2"))
    parser.add_argument("--pretrained", type=str2bool, default=cfg.get("pretrained", True))
    parser.add_argument("--image_size", type=int, default=cfg.get("image_size", 224))
    parser.add_argument("--feature_batch_size", type=int, default=cfg.get("feature_batch_size", 64))
    parser.add_argument("--feature_num_workers", type=int, default=cfg.get("feature_num_workers", 8))
    parser.add_argument("--feature_dtype", type=str, default=cfg.get("feature_dtype", "float16"))
    parser.add_argument("--amp", type=str2bool, default=cfg.get("amp", True))
    parser.add_argument("--force_features", type=str2bool, default=cfg.get("force_features", False))

    parser.add_argument("--k", type=int, default=cfg.get("k", 50))
    parser.add_argument("--topk_pool", type=int, default=cfg.get("topk_pool", 5))
    parser.add_argument("--max_patches_for_discovery", type=int, default=cfg.get("max_patches_for_discovery", 200000))
    parser.add_argument("--max_patches_per_image", type=int, default=cfg.get("max_patches_per_image", 64))
    parser.add_argument("--kmeans_iters", type=int, default=cfg.get("kmeans_iters", 25))
    parser.add_argument("--kmeans_batch_size", type=int, default=cfg.get("kmeans_batch_size", 8192))
    parser.add_argument("--activation_batch_size", type=int, default=cfg.get("activation_batch_size", 128))
    parser.add_argument("--pca_dim", type=int, default=cfg.get("pca_dim", 64))
    parser.add_argument("--pca_sample_size", type=int, default=cfg.get("pca_sample_size", 50000))
    parser.add_argument("--gmm_iters", type=int, default=cfg.get("gmm_iters", 12))
    parser.add_argument("--artifact_corr_threshold", type=float, default=cfg.get("artifact_corr_threshold", 0.2))

    parser.add_argument("--epochs", type=int, default=cfg.get("epochs", 160))
    parser.add_argument("--mlp_epochs", type=int, default=cfg.get("mlp_epochs", 120))
    parser.add_argument("--train_batch_size", type=int, default=cfg.get("train_batch_size", 256))
    parser.add_argument("--lr", type=float, default=cfg.get("lr", 1e-3))
    parser.add_argument("--weight_decay", type=float, default=cfg.get("weight_decay", 1e-4))
    parser.add_argument("--l1", type=float, default=cfg.get("l1", 1e-4))
    parser.add_argument("--mlp_hidden", type=int, default=cfg.get("mlp_hidden", 256))
    parser.add_argument("--mlp_dropout", type=float, default=cfg.get("mlp_dropout", 0.2))
    parser.add_argument("--class_weight", type=str2bool, default=cfg.get("class_weight", True))
    parser.add_argument(
        "--class_weight_mode",
        type=str,
        default=cfg.get("class_weight_mode", "inverse"),
        choices=["none", "inverse", "sqrt_inverse", "effective_num"],
    )
    parser.add_argument("--effective_num_beta", type=float, default=cfg.get("effective_num_beta", 0.999))
    parser.add_argument("--early_stop_patience", type=int, default=cfg.get("early_stop_patience", 30))
    parser.add_argument("--refit_train_val", type=str2bool, default=cfg.get("refit_train_val", False))

    args = parser.parse_args()
    args.models = [m.strip().upper() for m in str(args.models).split(",") if m.strip()]
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_image_path(row: pd.Series, image_dir: Path) -> Path:
    lesion_id = row.get("lesion_id")
    isic_id = row.get("isic_id")
    image_path = row.get("image_path")

    if pd.notna(lesion_id) and pd.notna(isic_id):
        nested = image_dir / str(lesion_id) / f"{isic_id}.jpg"
        if nested.exists():
            return nested

    if pd.notna(isic_id):
        flat = image_dir / f"{isic_id}.jpg"
        if flat.exists():
            return flat

    if pd.notna(image_path):
        custom = image_dir / str(image_path)
        if custom.exists():
            return custom

    return image_dir / str(lesion_id) / f"{isic_id}.jpg"


def load_milk10k_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    metadata = pd.read_csv(args.metadata_csv)
    groundtruth = pd.read_csv(args.groundtruth_csv)
    df = metadata.loc[metadata["image_type"].eq("dermoscopic")].copy()
    df = df.merge(groundtruth, on="lesion_id", how="inner")

    supplement_path = Path(args.supplement_csv)
    if supplement_path.exists():
        supplement = pd.read_csv(supplement_path)
        df = df.merge(supplement, on="isic_id", how="left")

    if not all(col in df.columns for col in LABEL_COLS):
        missing = sorted(set(LABEL_COLS) - set(df.columns))
        raise ValueError(f"Ground-truth columns are missing: {missing}")

    df["diagnosis_idx"] = df[LABEL_COLS].to_numpy(dtype=np.float32).argmax(axis=1).astype(int)
    df["diagnosis"] = [LABEL_COLS[i] for i in df["diagnosis_idx"].tolist()]
    image_dir = Path(args.image_dir)
    df["resolved_image_path"] = [str(resolve_image_path(row, image_dir)) for _, row in df.iterrows()]
    df["image_exists"] = df["resolved_image_path"].map(lambda p: Path(p).exists())
    missing_images = int((~df["image_exists"]).sum())
    if missing_images:
        print(f"Warning: {missing_images} dermoscopic images are missing and will use black placeholders.")

    if args.limit and args.limit > 0 and args.limit < len(df):
        df = stratified_limit(df, args.limit, args.seed)

    df = df.sort_values(["lesion_id", "isic_id"]).reset_index(drop=True)
    return df


def stratified_limit(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    pieces = []
    rng = np.random.default_rng(seed)
    counts = df["diagnosis_idx"].value_counts().sort_index()
    allocated = {}
    for cls, count in counts.items():
        allocated[int(cls)] = max(1, int(round(limit * count / len(df))))

    diff = limit - sum(allocated.values())
    class_order = list(counts.sort_values(ascending=False).index.astype(int))
    while diff != 0:
        for cls in class_order:
            if diff == 0:
                break
            if diff > 0:
                allocated[cls] += 1
                diff -= 1
            elif allocated[cls] > 1:
                allocated[cls] -= 1
                diff += 1

    for cls, n_cls in allocated.items():
        part = df.loc[df["diagnosis_idx"].eq(cls)]
        n_take = min(n_cls, len(part))
        sampled_idx = rng.choice(part.index.to_numpy(), size=n_take, replace=False)
        pieces.append(df.loc[sampled_idx])
    return pd.concat(pieces, axis=0).reset_index(drop=True)


def stratified_group_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group_labels = (
        df.groupby("lesion_id")["diagnosis_idx"]
        .agg(lambda values: int(pd.Series(values).mode().iloc[0]))
        .reset_index()
    )

    split_by_group: dict[str, str] = {}
    for cls, cls_groups in group_labels.groupby("diagnosis_idx"):
        groups = cls_groups["lesion_id"].to_numpy()
        rng.shuffle(groups)
        n = len(groups)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.10))
        if n >= 3:
            n_train = max(1, min(n - 2, n_train))
            n_val = max(1, min(n - n_train - 1, n_val))
        else:
            n_train = max(1, min(n, n_train))
            n_val = 0
        train_groups = set(groups[:n_train])
        val_groups = set(groups[n_train : n_train + n_val])
        for group in groups:
            if group in train_groups:
                split_by_group[str(group)] = "train"
            elif group in val_groups:
                split_by_group[str(group)] = "val"
            else:
                split_by_group[str(group)] = "test"

    out = df.copy()
    out["split"] = out["lesion_id"].map(lambda lesion_id: split_by_group[str(lesion_id)])
    return out


class MILK10KImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int):
        self.df = df.reset_index(drop=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor]:
        path = self.df.iloc[idx]["resolved_image_path"]
        try:
            image = Image.open(path).convert("RGB")
        except (FileNotFoundError, OSError):
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))
        return idx, self.transform(image)


def make_output_dir(base: str) -> Path:
    output_dir = Path(base)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def feature_cache_path(args: argparse.Namespace, output_dir: Path, n_rows: int) -> Path:
    limit_tag = f"limit{args.limit}" if args.limit else "full"
    dtype_tag = args.feature_dtype.lower()
    filename = f"features_{args.dinov2_model}_img{args.image_size}_{limit_tag}_n{n_rows}_{dtype_tag}.pt"
    return output_dir / "features" / filename


def build_dinov2(args: argparse.Namespace, device: torch.device) -> nn.Module:
    model = timm.create_model(
        args.dinov2_model,
        pretrained=args.pretrained,
        num_classes=0,
        img_size=args.image_size,
    )
    model.eval()
    model.to(device)
    return model


def extract_or_load_features(
    args: argparse.Namespace,
    df: pd.DataFrame,
    output_dir: Path,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    cache_path = feature_cache_path(args, output_dir, len(df))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not args.force_features:
        print(f"Loading cached DINOv2 features: {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    print("Extracting DINOv2 global features and patch tokens.")
    dataset = MILK10KImageDataset(df, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.feature_batch_size,
        shuffle=False,
        num_workers=args.feature_num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_dinov2(args, device)

    storage_dtype = torch.float16 if args.feature_dtype.lower() == "float16" else torch.float32
    globals_cpu: torch.Tensor | None = None
    patches_cpu: torch.Tensor | None = None

    autocast_enabled = bool(args.amp and device.type == "cuda")
    with torch.inference_mode():
        for indices, images in tqdm(loader, desc="DINOv2 feature extraction", unit="batch"):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=autocast_enabled):
                tokens = model.forward_features(images)
                if isinstance(tokens, dict):
                    tokens = tokens.get("x_norm_patchtokens", tokens.get("x", None))
                    if tokens is None:
                        raise ValueError("Unsupported timm DINOv2 feature dict.")
                if tokens.ndim != 3:
                    raise ValueError(f"Expected [B, tokens, dim] features, got {tuple(tokens.shape)}")
                global_features = model.forward_head(tokens, pre_logits=True)
                patch_tokens = tokens[:, 1:, :]

            if globals_cpu is None:
                n = len(df)
                dim = int(global_features.shape[-1])
                num_patches = int(patch_tokens.shape[1])
                globals_cpu = torch.empty((n, dim), dtype=storage_dtype)
                patches_cpu = torch.empty((n, num_patches, dim), dtype=storage_dtype)

            idx = indices.long()
            globals_cpu[idx] = global_features.detach().cpu().to(storage_dtype)
            patches_cpu[idx] = patch_tokens.detach().cpu().to(storage_dtype)

    payload = {
        "global_features": globals_cpu,
        "patch_tokens": patches_cpu,
        "label_cols": LABEL_COLS,
        "model": args.dinov2_model,
        "image_size": args.image_size,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    torch.save(payload, cache_path)
    print(f"Saved feature cache: {cache_path}")
    return payload


@dataclass
class FeatureScaler:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, x: torch.Tensor) -> "FeatureScaler":
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
        return cls(mean=mean, std=std)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class LinearClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainedClassifier:
    model: nn.Module
    scaler: FeatureScaler
    best_val_macro_f1: float
    best_epoch: int
    train_history: list[dict[str, float]]


def class_weights_from_labels(
    y: torch.Tensor,
    num_classes: int,
    device: torch.device,
    mode: str,
    effective_num_beta: float,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = torch.bincount(y.cpu(), minlength=num_classes).float().clamp_min(1.0)
    if mode == "sqrt_inverse":
        raw = 1.0 / torch.sqrt(counts)
    elif mode == "effective_num":
        beta = float(effective_num_beta)
        raw = (1.0 - beta) / (1.0 - torch.pow(torch.full_like(counts, beta), counts))
    else:
        raw = 1.0 / counts
    weights = raw / raw.sum() * num_classes
    return weights.to(device)


def build_classifier_model(
    model_type: str,
    in_dim: int,
    num_classes: int,
    args: argparse.Namespace,
) -> tuple[nn.Module, int]:
    if model_type == "mlp":
        return MLPClassifier(in_dim, args.mlp_hidden, num_classes, args.mlp_dropout), args.mlp_epochs
    return LinearClassifier(in_dim, num_classes), args.epochs


def fit_classifier_epochs(
    model: nn.Module,
    x_train_s: torch.Tensor,
    y_train: torch.Tensor,
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
    epochs: int,
    l1: float,
) -> nn.Module:
    model.to(device)
    weights = (
        class_weights_from_labels(y_train, num_classes, device, args.class_weight_mode, args.effective_num_beta)
        if args.class_weight
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    y_train_device = y_train.to(device)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(x_train_s))
        for start in range(0, len(perm), args.train_batch_size):
            batch_idx = perm[start : start + args.train_batch_size]
            xb = x_train_s[batch_idx].to(device)
            yb = y_train_device[batch_idx]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            if l1 > 0:
                if isinstance(model, LinearClassifier):
                    loss = loss + l1 * model.linear.weight.abs().sum()
                else:
                    loss = loss + l1 * sum(param.abs().sum() for name_, param in model.named_parameters() if "weight" in name_)
            loss.backward()
            optimizer.step()
    return model


def train_classifier(
    name: str,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
    model_type: str,
    l1: float,
) -> TrainedClassifier:
    scaler = FeatureScaler.fit(x_train.float())
    x_train_s = scaler.transform(x_train.float())
    x_val_s = scaler.transform(x_val.float())

    model, epochs = build_classifier_model(model_type, x_train_s.shape[1], num_classes, args)
    model.to(device)

    weights = (
        class_weights_from_labels(y_train, num_classes, device, args.class_weight_mode, args.effective_num_beta)
        if args.class_weight
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    y_train_device = y_train.to(device)
    y_val_np = y_val.cpu().numpy()
    best_score = -1.0
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    patience_count = 0
    history: list[dict[str, float]] = []

    for epoch in tqdm(range(epochs), desc=f"Training {name}", leave=False):
        model.train()
        perm = torch.randperm(len(x_train_s))
        total_loss = 0.0
        for start in range(0, len(perm), args.train_batch_size):
            batch_idx = perm[start : start + args.train_batch_size]
            xb = x_train_s[batch_idx].to(device)
            yb = y_train_device[batch_idx]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            if l1 > 0:
                if isinstance(model, LinearClassifier):
                    loss = loss + l1 * model.linear.weight.abs().sum()
                else:
                    loss = loss + l1 * sum(param.abs().sum() for name_, param in model.named_parameters() if "weight" in name_)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_idx)

        probs_val = predict_probs(model, scaler, x_val_s, device, already_scaled=True, move_to_cpu=False)
        val_metrics = classification_metrics(y_val_np, probs_val, LABEL_COLS)
        val_macro = float(val_metrics["macro_f1_argmax"])
        history.append({"epoch": epoch + 1, "loss": total_loss / len(perm), "val_macro_f1": val_macro})
        if val_macro > best_score:
            best_score = val_macro
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= args.early_stop_patience:
                break

    model.load_state_dict(best_state)
    if args.refit_train_val:
        x_joint = torch.cat([x_train.float(), x_val.float()], dim=0)
        y_joint = torch.cat([y_train, y_val], dim=0)
        scaler = FeatureScaler.fit(x_joint)
        x_joint_s = scaler.transform(x_joint)
        model, _ = build_classifier_model(model_type, x_joint_s.shape[1], num_classes, args)
        model = fit_classifier_epochs(model, x_joint_s, y_joint, num_classes, args, device, best_epoch, l1)

    return TrainedClassifier(
        model=model.cpu(),
        scaler=scaler,
        best_val_macro_f1=best_score,
        best_epoch=best_epoch,
        train_history=history,
    )


def predict_probs(
    model: nn.Module,
    scaler: FeatureScaler,
    x: torch.Tensor,
    device: torch.device,
    already_scaled: bool = False,
    batch_size: int = 4096,
    move_to_cpu: bool = True,
) -> np.ndarray:
    model = model.to(device)
    model.eval()
    if already_scaled:
        x_scaled = x.float()
    else:
        x_scaled = scaler.transform(x.float())
    probs = []
    with torch.inference_mode():
        for start in range(0, len(x_scaled), batch_size):
            xb = x_scaled[start : start + batch_size].to(device)
            probs.append(torch.softmax(model(xb), dim=1).detach().cpu())
    if move_to_cpu:
        model.cpu()
    return torch.cat(probs, dim=0).numpy()


def classification_metrics(y_true: np.ndarray, probs: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64)
    num_classes = len(class_names)
    y_pred = probs.argmax(axis=1)

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred, strict=False):
        if 0 <= true < num_classes and 0 <= pred < num_classes:
            confusion[true, pred] += 1

    per_class_f1 = {}
    per_class_recall = {}
    f1s = []
    recalls = []
    for idx, name in enumerate(class_names):
        tp = confusion[idx, idx]
        fp = confusion[:, idx].sum() - tp
        fn = confusion[idx, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        per_class_f1[name] = float(f1)
        per_class_recall[name] = float(recall)
        f1s.append(f1)
        recalls.append(recall)

    onehot = np.eye(num_classes, dtype=bool)[y_true]
    threshold_preds = probs >= 0.5
    threshold_f1s = []
    for idx in range(num_classes):
        tp = np.logical_and(threshold_preds[:, idx], onehot[:, idx]).sum()
        fp = np.logical_and(threshold_preds[:, idx], ~onehot[:, idx]).sum()
        fn = np.logical_and(~threshold_preds[:, idx], onehot[:, idx]).sum()
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        threshold_f1s.append(2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0)

    aucs = []
    for idx in range(num_classes):
        auc = binary_auc(onehot[:, idx].astype(int), probs[:, idx])
        if not math.isnan(auc):
            aucs.append(auc)

    malignant_idx = [i for i, name in enumerate(class_names) if name in MALIGNANT_CLASSES]
    true_malignant = np.isin(y_true, malignant_idx)
    pred_malignant = np.isin(y_pred, malignant_idx)
    if true_malignant.sum() > 0:
        malignancy_recall = float(np.logical_and(true_malignant, pred_malignant).sum() / true_malignant.sum())
    else:
        malignancy_recall = float("nan")

    return {
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1_argmax": float(np.mean(f1s)),
        "official_macro_f1_threshold_0_5": float(np.mean(threshold_f1s)),
        "macro_auroc": float(np.mean(aucs)) if aucs else float("nan"),
        "balanced_accuracy": float(np.mean(recalls)),
        "ece": expected_calibration_error(y_true, probs),
        "malignancy_group_recall": malignancy_recall,
        "per_class_f1": per_class_f1,
        "per_class_recall": per_class_recall,
    }


def binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    pos_rank_sum = ranks[y_true == 1].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        if hi == 1.0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def select_indices(df: pd.DataFrame, split: str) -> np.ndarray:
    return df.index[df["split"].eq(split)].to_numpy(dtype=np.int64).copy()


def monet_features(df: pd.DataFrame) -> torch.Tensor:
    values = df[MONET_COLS].fillna(df[MONET_COLS].median(numeric_only=True)).fillna(0.0)
    return torch.tensor(values.to_numpy(dtype=np.float32))


def sample_train_patches(
    patch_tokens: torch.Tensor,
    train_idx: np.ndarray,
    max_patches: int,
    max_patches_per_image: int,
    seed: int,
) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    train_patches = patch_tokens[train_idx].float()
    n_img, n_patch, dim = train_patches.shape
    per_image = min(max_patches_per_image, n_patch)
    patches = []
    for i in range(n_img):
        perm = torch.randperm(n_patch, generator=gen)[:per_image]
        patches.append(train_patches[i, perm])
    sampled = torch.cat(patches, dim=0)
    if len(sampled) > max_patches:
        idx = torch.randperm(len(sampled), generator=gen)[:max_patches]
        sampled = sampled[idx]
    return sampled.contiguous()


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=1, eps=1e-6)


def mini_batch_kmeans(
    x: torch.Tensor,
    k: int,
    n_iter: int,
    batch_size: int,
    spherical: bool,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    x = x.float().contiguous()
    n, dim = x.shape
    if n < k:
        raise ValueError(f"Need at least k samples for k-means, got n={n}, k={k}")
    gen = torch.Generator().manual_seed(seed)
    init_idx = torch.randperm(n, generator=gen)[:k]
    centroids = x[init_idx].to(device)
    if spherical:
        centroids = normalize_rows(centroids)

    for _ in tqdm(range(n_iter), desc="k-means", leave=False):
        sums = torch.zeros((k, dim), dtype=torch.float32, device=device)
        counts = torch.zeros(k, dtype=torch.float32, device=device)
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch = x[idx].to(device)
            if spherical:
                batch = normalize_rows(batch)
                scores = batch @ centroids.T
                labels = scores.argmax(dim=1)
            else:
                distances = torch.cdist(batch, centroids, p=2).pow(2)
                labels = distances.argmin(dim=1)
            sums.index_add_(0, labels, batch)
            counts += torch.bincount(labels, minlength=k).float()

        non_empty = counts > 0
        updated = centroids.clone()
        updated[non_empty] = sums[non_empty] / counts[non_empty].unsqueeze(1)
        if spherical:
            updated = normalize_rows(updated)
        centroids = updated
    return centroids.detach().cpu()


@dataclass
class ConceptModel:
    method: str
    centroids: torch.Tensor | None = None
    mean: torch.Tensor | None = None
    std: torch.Tensor | None = None
    pca_mean: torch.Tensor | None = None
    pca_components: torch.Tensor | None = None
    gmm_means: torch.Tensor | None = None
    gmm_vars: torch.Tensor | None = None
    gmm_weights: torch.Tensor | None = None
    fit_stats: dict[str, Any] | None = None


def fit_euclidean_kmeans(sample: torch.Tensor, args: argparse.Namespace, device: torch.device) -> ConceptModel:
    mean = sample.mean(dim=0, keepdim=True)
    std = sample.std(dim=0, keepdim=True).clamp_min(1e-6)
    x = (sample - mean) / std
    centroids = mini_batch_kmeans(
        x,
        args.k,
        args.kmeans_iters,
        args.kmeans_batch_size,
        spherical=False,
        seed=args.seed,
        device=device,
    )
    return ConceptModel(method="euclidean_kmeans", centroids=centroids, mean=mean, std=std)


def fit_spherical_kmeans(sample: torch.Tensor, args: argparse.Namespace, device: torch.device) -> ConceptModel:
    x = normalize_rows(sample)
    centroids = mini_batch_kmeans(
        x,
        args.k,
        args.kmeans_iters,
        args.kmeans_batch_size,
        spherical=True,
        seed=args.seed + 1,
        device=device,
    )
    return ConceptModel(method="spherical_kmeans", centroids=centroids)


def fit_medoid_prototypes(sample: torch.Tensor, args: argparse.Namespace, device: torch.device) -> ConceptModel:
    x = normalize_rows(sample)
    centroids = mini_batch_kmeans(
        x,
        args.k,
        args.kmeans_iters,
        args.kmeans_batch_size,
        spherical=True,
        seed=args.seed + 2,
        device=device,
    )
    medoids = []
    centroids_device = centroids.to(device)
    for start in range(0, len(x), args.kmeans_batch_size):
        batch = x[start : start + args.kmeans_batch_size].to(device)
        sims = batch @ centroids_device.T
        if start == 0:
            best_scores = sims.max(dim=0).values
            best_indices = sims.argmax(dim=0) + start
        else:
            batch_scores = sims.max(dim=0).values
            batch_indices = sims.argmax(dim=0) + start
            replace = batch_scores > best_scores
            best_scores[replace] = batch_scores[replace]
            best_indices[replace] = batch_indices[replace]
    for idx in best_indices.cpu().tolist():
        medoids.append(x[idx])
    return ConceptModel(method="medoid_prototype", centroids=torch.stack(medoids, dim=0))


def fit_gmm_pca(sample: torch.Tensor, args: argparse.Namespace, device: torch.device) -> ConceptModel:
    gen = torch.Generator().manual_seed(args.seed + 3)
    x = sample.float()
    if len(x) > args.pca_sample_size:
        pca_idx = torch.randperm(len(x), generator=gen)[: args.pca_sample_size]
        x_for_pca = x[pca_idx]
    else:
        x_for_pca = x

    pca_mean = x_for_pca.mean(dim=0, keepdim=True)
    centered = x_for_pca - pca_mean
    cov = centered.T @ centered / max(1, len(centered) - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    pca_dim = min(args.pca_dim, eigvecs.shape[1])
    components = eigvecs[:, order[:pca_dim]].contiguous()
    z = (x - pca_mean) @ components

    init_means = mini_batch_kmeans(
        z,
        args.k,
        max(3, args.kmeans_iters // 2),
        args.kmeans_batch_size,
        spherical=False,
        seed=args.seed + 4,
        device=device,
    ).to(device)
    means, vars_, weights = fit_diag_gmm(
        z,
        init_means,
        args.gmm_iters,
        args.kmeans_batch_size,
        device,
    )
    return ConceptModel(
        method="gmm_pca",
        pca_mean=pca_mean,
        pca_components=components,
        gmm_means=means.cpu(),
        gmm_vars=vars_.cpu(),
        gmm_weights=weights.cpu(),
    )


def fit_diag_gmm(
    z: torch.Tensor,
    init_means: torch.Tensor,
    n_iter: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = z.float().contiguous()
    n, dim = z.shape
    k = init_means.shape[0]
    means = init_means.float().to(device)
    global_var = z.var(dim=0, keepdim=True).clamp_min(1e-4).to(device)
    vars_ = global_var.repeat(k, 1)
    weights = torch.full((k,), 1.0 / k, dtype=torch.float32, device=device)

    for _ in tqdm(range(n_iter), desc="diag GMM", leave=False):
        resp_sum = torch.zeros(k, dtype=torch.float32, device=device)
        x_sum = torch.zeros((k, dim), dtype=torch.float32, device=device)
        x2_sum = torch.zeros((k, dim), dtype=torch.float32, device=device)
        for start in range(0, n, batch_size):
            batch = z[start : start + batch_size].to(device)
            logp = diag_gmm_log_prob(batch, means, vars_, weights)
            resp = torch.softmax(logp, dim=1)
            resp_sum += resp.sum(dim=0)
            x_sum += resp.T @ batch
            x2_sum += resp.T @ (batch * batch)
        resp_safe = resp_sum.clamp_min(1e-6)
        means = x_sum / resp_safe.unsqueeze(1)
        vars_ = (x2_sum / resp_safe.unsqueeze(1) - means * means).clamp_min(1e-4)
        weights = resp_safe / resp_safe.sum()
    return means, vars_, weights


def diag_gmm_log_prob(
    x: torch.Tensor,
    means: torch.Tensor,
    vars_: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    x_exp = x.unsqueeze(1)
    mean_exp = means.unsqueeze(0)
    var_exp = vars_.unsqueeze(0)
    mahal = ((x_exp - mean_exp).pow(2) / var_exp).sum(dim=2)
    log_det = vars_.log().sum(dim=1).unsqueeze(0)
    log_weights = weights.clamp_min(1e-8).log().unsqueeze(0)
    return -0.5 * (mahal + log_det + x.shape[1] * math.log(2.0 * math.pi)) + log_weights


def concept_activation_vectors(
    patch_tokens: torch.Tensor,
    concept_model: ConceptModel,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    n = patch_tokens.shape[0]
    k = args.k
    activations = torch.empty((n, k), dtype=torch.float32)
    topk = min(args.topk_pool, patch_tokens.shape[1])

    if concept_model.method == "euclidean_kmeans":
        mean = concept_model.mean.to(device)
        std = concept_model.std.to(device)
        centroids = concept_model.centroids.to(device)
    elif concept_model.method in {"spherical_kmeans", "medoid_prototype"}:
        centroids = normalize_rows(concept_model.centroids.to(device))
    elif concept_model.method == "gmm_pca":
        pca_mean = concept_model.pca_mean.to(device)
        pca_components = concept_model.pca_components.to(device)
        gmm_means = concept_model.gmm_means.to(device)
        gmm_vars = concept_model.gmm_vars.to(device)
        gmm_weights = concept_model.gmm_weights.to(device)
    else:
        raise ValueError(f"Unknown concept method: {concept_model.method}")

    with torch.inference_mode():
        for start in tqdm(range(0, n, args.activation_batch_size), desc=f"Activating {concept_model.method}", leave=False):
            batch = patch_tokens[start : start + args.activation_batch_size].float().to(device)
            b, p, d = batch.shape
            flat = batch.reshape(b * p, d)
            if concept_model.method == "euclidean_kmeans":
                flat = (flat - mean) / std
                distances = torch.cdist(flat, centroids, p=2).pow(2)
                scores = -distances / math.sqrt(flat.shape[1])
            elif concept_model.method in {"spherical_kmeans", "medoid_prototype"}:
                flat = normalize_rows(flat)
                scores = flat @ centroids.T
            else:
                z = (flat - pca_mean) @ pca_components
                scores = diag_gmm_log_prob(z, gmm_means, gmm_vars, gmm_weights)
            scores = scores.reshape(b, p, k)
            pooled = scores.topk(topk, dim=1).values.mean(dim=1)
            activations[start : start + b] = pooled.cpu()
    return activations


def concept_quality(
    activations: torch.Tensor,
    y: np.ndarray,
    df: pd.DataFrame,
    concept_model: ConceptModel,
    args: argparse.Namespace,
) -> dict[str, Any]:
    act_np = activations.numpy()
    v = np.abs(act_np)
    l1 = v.sum(axis=1)
    l2 = np.sqrt((v * v).sum(axis=1)).clip(min=1e-8)
    k = act_np.shape[1]
    sparsity = ((math.sqrt(k) - l1 / l2) / max(1e-8, math.sqrt(k) - 1.0)).mean()

    class_means = []
    for cls in range(len(LABEL_COLS)):
        mask = y == cls
        if mask.any():
            class_means.append(act_np[mask].mean(axis=0))
    if class_means:
        class_means_np = np.stack(class_means, axis=0)
        selectivity = (class_means_np.max(axis=0) - class_means_np.min(axis=0)).mean()
    else:
        selectivity = float("nan")

    monet = df[MONET_COLS].to_numpy(dtype=np.float32)
    corr = correlation_matrix(act_np, monet)
    abs_corr = np.abs(corr)
    best_idx = abs_corr.argmax(axis=1)
    best_corr = abs_corr.max(axis=1)
    best_monet = [MONET_COLS[i] for i in best_idx.tolist()]
    artifact_mask = [
        monet_name in ARTIFACT_MONET_COLS and best_corr[i] >= args.artifact_corr_threshold
        for i, monet_name in enumerate(best_monet)
    ]

    diversity = float("nan")
    if concept_model.centroids is not None:
        centroids = concept_model.centroids.float()
        centroids = normalize_rows(centroids)
        sims = torch.abs(centroids @ centroids.T).numpy()
        upper = sims[np.triu_indices_from(sims, k=1)]
        diversity = float(1.0 - upper.mean()) if len(upper) else float("nan")

    return {
        "coherence_proxy_top_activation_mean": float(np.mean(np.max(act_np, axis=1))),
        "diversity_proxy_1_minus_abs_centroid_cosine": diversity,
        "sparsity_hoyer": float(sparsity),
        "class_selectivity_mean_range": float(selectivity),
        "monet_alignment_mean_best_abs_corr": float(best_corr.mean()),
        "monet_alignment_max_abs_corr": float(best_corr.max()),
        "artifact_concept_rate_proxy": float(np.mean(artifact_mask)),
        "artifact_concepts": [
            {"concept": int(i), "best_monet": best_monet[i], "abs_corr": float(best_corr[i])}
            for i, is_artifact in enumerate(artifact_mask)
            if is_artifact
        ],
        "stability": None,
    }


def correlation_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    a_std = a.std(axis=0, keepdims=True)
    b_std = b.std(axis=0, keepdims=True)
    a = a / np.clip(a_std, 1e-8, None)
    b = b / np.clip(b_std, 1e-8, None)
    return (a.T @ b) / max(1, len(a) - 1)


def top_concept_ablation(
    trained: TrainedClassifier,
    x_raw: torch.Tensor,
    device: torch.device,
    top_ms: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    if not isinstance(trained.model, LinearClassifier):
        return {}
    model = trained.model
    scaler = trained.scaler
    x_raw = x_raw.float()
    x_std = scaler.transform(x_raw)
    zero_raw = torch.zeros_like(x_raw[:1])
    zero_std = scaler.transform(zero_raw)[0]

    weight = model.linear.weight.detach().cpu()
    bias = model.linear.bias.detach().cpu()
    logits = x_std @ weight.T + bias
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1)

    out: dict[str, float] = {}
    for m in top_ms:
        drops = []
        for i in range(len(x_std)):
            cls = int(pred[i])
            contribution = weight[cls] * x_std[i]
            top_idx = torch.argsort(contribution, descending=True)[: min(m, x_std.shape[1])]
            ablated = x_std[i].clone()
            ablated[top_idx] = zero_std[top_idx]
            ablated_logits = ablated @ weight.T + bias
            ablated_probs = torch.softmax(ablated_logits, dim=0)
            drops.append(float(probs[i, cls] - ablated_probs[cls]))
        out[f"drop_top_{m}_mean"] = float(np.mean(drops))
    return out


def subgroup_metrics(df: pd.DataFrame, y_true: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if "skin_tone_class" not in df.columns:
        return result
    for subgroup, part in df.groupby("skin_tone_class", dropna=False):
        idx = part.index.to_numpy(dtype=np.int64)
        if len(idx) < 5:
            continue
        metrics = classification_metrics(y_true[idx], probs[idx], LABEL_COLS)
        result[str(subgroup)] = {
            "n": int(len(idx)),
            "macro_f1_argmax": metrics["macro_f1_argmax"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "malignancy_group_recall": metrics["malignancy_group_recall"],
        }
    return result


def run_experiment() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = make_output_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = load_milk10k_dataframe(args)
    df = stratified_group_split(df, args.seed)
    df.to_csv(output_dir / "milk10k_dermoscopy_split.csv", index=False)
    split_counts = df.groupby(["split", "diagnosis"]).size().unstack(fill_value=0)
    split_counts.to_csv(output_dir / "split_class_counts.csv")
    print(split_counts)

    features = extract_or_load_features(args, df, output_dir, device)
    global_features = features["global_features"].float()
    patch_tokens = features["patch_tokens"]
    y = torch.tensor(df["diagnosis_idx"].to_numpy(dtype=np.int64))
    y_np = y.numpy()

    train_idx = select_indices(df, "train")
    val_idx = select_indices(df, "val")
    test_idx = select_indices(df, "test")
    train_val_idx = np.concatenate([train_idx, val_idx])

    results: dict[str, Any] = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "device": str(device),
        "n_samples": int(len(df)),
        "split_counts": split_counts.to_dict(),
        "experiments": {},
    }

    def train_and_eval(
        exp_id: str,
        display_name: str,
        x_all: torch.Tensor,
        model_type: str,
        sparse_l1: float,
        concept_backed: bool,
        concept_model: ConceptModel | None = None,
    ) -> None:
        print(f"\nRunning {exp_id}: {display_name}")
        trained = train_classifier(
            display_name,
            x_all[train_idx],
            y[train_idx],
            x_all[val_idx],
            y[val_idx],
            len(LABEL_COLS),
            args,
            device,
            model_type,
            sparse_l1,
        )
        probs_all = predict_probs(trained.model, trained.scaler, x_all, device)
        test_metrics = classification_metrics(y_np[test_idx], probs_all[test_idx], LABEL_COLS)
        val_metrics = classification_metrics(y_np[val_idx], probs_all[val_idx], LABEL_COLS)
        ablation = top_concept_ablation(trained, x_all[test_idx], device) if concept_backed else {}
        quality = concept_quality(x_all, y_np, df, concept_model, args) if concept_model is not None else {}
        subgroups = subgroup_metrics(df.iloc[test_idx].reset_index(drop=True), y_np[test_idx], probs_all[test_idx])

        result = {
            "name": display_name,
            "best_val_macro_f1": trained.best_val_macro_f1,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "top_concept_ablation": ablation,
            "concept_quality": quality,
            "subgroup_skin_tone_test": subgroups,
            "best_epoch": trained.best_epoch,
            "history_tail": trained.train_history[-10:],
        }
        results["experiments"][exp_id] = result
        save_json(results, output_dir / "results.json")

    if "E1" in args.models:
        train_and_eval("E1", "B1 black-box linear probe", global_features, "linear", 0.0, False)

    if "E2" in args.models:
        train_and_eval("E2", "B1 black-box MLP", global_features, "mlp", 0.0, False)

    if "E3" in args.models:
        train_and_eval("E3", "B2 pre-defined MONET sparse logistic", monet_features(df), "linear", args.l1, True)

    discovery_models = {
        "E4": ("CD-1 Euclidean k-means", fit_euclidean_kmeans),
        "E5": ("CD-2 spherical k-means", fit_spherical_kmeans),
        "E6": ("CD-3 GMM + PCA", fit_gmm_pca),
        "E7": ("CD-4 medoid prototype", fit_medoid_prototypes),
    }
    selected_discovery = [exp_id for exp_id in discovery_models if exp_id in args.models or "E8" in args.models]
    activation_bank: dict[str, torch.Tensor] = {}
    if selected_discovery:
        patch_sample = sample_train_patches(
            patch_tokens,
            train_idx,
            args.max_patches_for_discovery,
            args.max_patches_per_image,
            args.seed,
        )
        print(f"Discovery patch sample: {tuple(patch_sample.shape)}")
        for exp_id in selected_discovery:
            display_name, fitter = discovery_models[exp_id]
            print(f"\nFitting concepts for {exp_id}: {display_name}")
            concept_model = fitter(patch_sample, args, device)
            activations = concept_activation_vectors(patch_tokens, concept_model, args, device)
            activation_bank[exp_id] = activations
            activation_path = output_dir / f"activations_{exp_id}_{concept_model.method}_k{args.k}.pt"
            torch.save({"activations": activations, "method": concept_model.method}, activation_path)
            if exp_id in args.models:
                train_and_eval(exp_id, display_name, activations, "linear", args.l1, True, concept_model)

    if "E8" in args.models:
        hybrid_parts = [monet_features(df)]
        for exp_id in ["E4", "E5", "E6", "E7"]:
            if exp_id in activation_bank:
                hybrid_parts.append(activation_bank[exp_id])
        hybrid_features = torch.cat(hybrid_parts, dim=1)
        train_and_eval(
            "E8",
            "Hybrid MONET + discovered concept bank",
            hybrid_features,
            "linear",
            args.l1,
            True,
            None,
        )

    save_json(results, output_dir / "results.json")
    write_summary(results, output_dir)
    print(f"\nDone. Results written to {output_dir}")


def save_json(payload: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, argparse.Namespace):
        return vars(value)
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_summary(results: dict[str, Any], output_dir: Path) -> None:
    rows = []
    for exp_id, result in results["experiments"].items():
        metrics = result["test_metrics"]
        quality = result.get("concept_quality", {})
        ablation = result.get("top_concept_ablation", {})
        rows.append(
            {
                "id": exp_id,
                "name": result["name"],
                "macro_f1_argmax": metrics["macro_f1_argmax"],
                "official_macro_f1_threshold_0_5": metrics["official_macro_f1_threshold_0_5"],
                "macro_auroc": metrics["macro_auroc"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "ece": metrics["ece"],
                "malignancy_group_recall": metrics["malignancy_group_recall"],
                "ablation_drop_top_1": ablation.get("drop_top_1_mean"),
                "ablation_drop_top_3": ablation.get("drop_top_3_mean"),
                "ablation_drop_top_5": ablation.get("drop_top_5_mean"),
                "monet_alignment": quality.get("monet_alignment_mean_best_abs_corr"),
                "artifact_rate": quality.get("artifact_concept_rate_proxy"),
                "sparsity": quality.get("sparsity_hoyer"),
            }
        )
    summary = pd.DataFrame(rows).sort_values("id")
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)

    lines = [
        "# MILK10K Concept Discovery Experiment Results",
        "",
        f"- Created at: {results['created_at']}",
        f"- Device: {results['device']}",
        f"- Samples: {results['n_samples']}",
        "",
        "## Test Metrics",
        "",
    ]
    if not summary.empty:
        lines.append(dataframe_to_markdown(summary))
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Split is dermoscopy-only lesion-id group split with class stratification (70/10/20).",
            "- DINOv2 features are extracted with timm ViT-S/14 and cached under the output directory.",
            "- Concept quality metrics are proxy metrics computed from activations and MONET correlations; human naming/coherence review still requires visual inspection of top examples.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    rows = []
    for _, row in df.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.4f}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        rows.append(values)

    def clean(cell: str) -> str:
        return cell.replace("|", "\\|")

    lines = [
        "| " + " | ".join(clean(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(clean(cell) for cell in row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    run_experiment()
