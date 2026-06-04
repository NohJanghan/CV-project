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


IGNORE_ANNOTATION_COLS = {"Unnamed: 0", "ImageID", "Do not consider this image"}


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"true", "t", "yes", "y", "1"}:
        return True
    if value in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def load_yaml(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--config_path", type=str, default=None)
    base_args, _ = base.parse_known_args()
    cfg = load_yaml(base_args.config_path)

    parser = argparse.ArgumentParser(description="Run SkinCon E1-E7 concept bottleneck experiments.")
    parser.add_argument("--config_path", type=str, default=base_args.config_path)
    parser.add_argument("--annotations_csv", type=str, default=cfg.get("annotations_csv", "data/skincon/annotations_fitzpatrick17k.csv"))
    parser.add_argument("--fitzpatrick_metadata_csv", type=str, default=cfg.get("fitzpatrick_metadata_csv", "data/skincon/fitzpatrick17k.csv"))
    parser.add_argument("--image_dir", type=str, default=cfg.get("image_dir", "data/skincon/fitzpatrick17k_images"))
    parser.add_argument("--output_dir", type=str, default=cfg.get("output_dir", "output/skincon_concept_experiment"))
    parser.add_argument("--seed", type=int, default=cfg.get("seed", 42))
    parser.add_argument("--min_target_count", type=int, default=cfg.get("min_target_count", 20))
    parser.add_argument("--models", type=str, default=cfg.get("models", "E1,E2,E3,E4,E5,E6,E7"))

    parser.add_argument("--dinov2_model", type=str, default=cfg.get("dinov2_model", "vit_small_patch14_dinov2"))
    parser.add_argument("--pretrained", type=str2bool, default=cfg.get("pretrained", True))
    parser.add_argument("--image_size", type=int, default=cfg.get("image_size", 224))
    parser.add_argument("--feature_batch_size", type=int, default=cfg.get("feature_batch_size", 64))
    parser.add_argument("--feature_num_workers", type=int, default=cfg.get("feature_num_workers", 4))
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

    parser.add_argument("--epochs", type=int, default=cfg.get("epochs", 160))
    parser.add_argument("--mlp_epochs", type=int, default=cfg.get("mlp_epochs", 120))
    parser.add_argument("--train_batch_size", type=int, default=cfg.get("train_batch_size", 256))
    parser.add_argument("--lr", type=float, default=cfg.get("lr", 1e-3))
    parser.add_argument("--weight_decay", type=float, default=cfg.get("weight_decay", 1e-4))
    parser.add_argument("--l1", type=float, default=cfg.get("l1", 1e-4))
    parser.add_argument("--mlp_hidden", type=int, default=cfg.get("mlp_hidden", 256))
    parser.add_argument("--mlp_dropout", type=float, default=cfg.get("mlp_dropout", 0.2))
    parser.add_argument("--class_weight_mode", type=str, default=cfg.get("class_weight_mode", "sqrt_inverse"), choices=["none", "inverse", "sqrt_inverse"])
    parser.add_argument("--early_stop_patience", type=int, default=cfg.get("early_stop_patience", 30))
    parser.add_argument("--refit_train_val", type=str2bool, default=cfg.get("refit_train_val", False))

    args = parser.parse_args()
    args.models = [item.strip().upper() for item in str(args.models).split(",") if item.strip()]
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def concept_columns(df: pd.DataFrame) -> list[str]:
    cols = [col for col in df.columns if col not in IGNORE_ANNOTATION_COLS]
    return [col for col in cols if pd.api.types.is_numeric_dtype(df[col])]


def resolve_image_path(row: pd.Series, image_dir: Path) -> Path:
    candidates = [
        image_dir / str(row["ImageID"]),
        image_dir / f"{row['md5hash']}.jpg",
        image_dir / f"{row['md5hash']}.jpeg",
        image_dir / f"{row['md5hash']}.png",
    ]
    url_name = row.get("url_alphanum")
    if pd.notna(url_name):
        candidates.append(image_dir / str(url_name))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_skincon_dataframe(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], list[str]]:
    annotations = pd.read_csv(args.annotations_csv)
    concepts = concept_columns(annotations)
    annotations = annotations.copy()
    annotations["md5hash"] = annotations["ImageID"].astype(str).str.replace(r"\.[^.]+$", "", regex=True)

    if "Do not consider this image" in annotations.columns:
        annotations = annotations.loc[annotations["Do not consider this image"].fillna(0).astype(int).eq(0)].copy()

    metadata = pd.read_csv(args.fitzpatrick_metadata_csv)
    df = annotations.merge(metadata, on="md5hash", how="inner", validate="many_to_one")
    if "label" not in df.columns:
        raise ValueError("Fitzpatrick17k metadata must contain a `label` diagnosis column.")

    for col in concepts:
        df[col] = df[col].fillna(0).astype(np.float32)

    df["target_name"] = df["label"].astype(str)
    counts = df["target_name"].value_counts()
    keep = counts[counts >= args.min_target_count].index
    df = df.loc[df["target_name"].isin(keep)].copy()
    target_names = sorted(df["target_name"].dropna().unique())
    target_to_idx = {name: idx for idx, name in enumerate(target_names)}
    df["target_idx"] = df["target_name"].map(target_to_idx).astype(int)

    image_dir = Path(args.image_dir)
    df["image_path"] = [str(resolve_image_path(row, image_dir)) for _, row in df.iterrows()]
    df["image_exists"] = df["image_path"].map(lambda path: Path(path).exists())
    missing = int((~df["image_exists"]).sum())
    if missing:
        examples = df.loc[~df["image_exists"], "image_path"].head(5).tolist()
        raise FileNotFoundError(
            f"{missing} SkinCon/Fitzpatrick17k image files are missing under {args.image_dir}. "
            f"Examples: {examples}"
        )

    return df.sort_values(["target_name", "ImageID"]).reset_index(drop=True), concepts, target_names


def stratified_image_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Image-level stratified split by Fitzpatrick17k diagnosis label.

    SkinCon/Fitzpatrick17k metadata does not provide a stable patient or lesion group
    identifier, so this split cannot enforce patient-level de-duplication.
    """
    rng = np.random.default_rng(seed)
    split = np.empty(len(df), dtype=object)
    for _, part in df.groupby("target_idx"):
        indices = part.index.to_numpy(dtype=np.int64).copy()
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.10))
        if n >= 3:
            n_train = max(1, min(n - 2, n_train))
            n_val = max(1, min(n - n_train - 1, n_val))
        else:
            n_train = max(1, n)
            n_val = 0
        split[indices[:n_train]] = "train"
        split[indices[n_train : n_train + n_val]] = "val"
        split[indices[n_train + n_val :]] = "test"
    out = df.copy()
    out["split"] = split
    return out


class SkinConImageDataset(Dataset):
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
        path = self.df.iloc[idx]["image_path"]
        image = Image.open(path).convert("RGB")
        return idx, self.transform(image)


def build_dinov2(args: argparse.Namespace, device: torch.device) -> nn.Module:
    model = timm.create_model(args.dinov2_model, pretrained=args.pretrained, num_classes=0, img_size=args.image_size)
    return model.eval().to(device)


def feature_cache_path(args: argparse.Namespace, output_dir: Path, n_rows: int) -> Path:
    return output_dir / "features" / f"features_{args.dinov2_model}_img{args.image_size}_n{n_rows}_{args.feature_dtype}.pt"


def extract_or_load_features(args: argparse.Namespace, df: pd.DataFrame, output_dir: Path, device: torch.device) -> dict[str, torch.Tensor]:
    cache_path = feature_cache_path(args, output_dir, len(df))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not args.force_features:
        print(f"Loading cached features: {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    dataset = SkinConImageDataset(df, args.image_size)
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
                features = model.forward_features(images)
                if isinstance(features, dict):
                    if "x_norm_clstoken" in features and "x_norm_patchtokens" in features:
                        global_features = features["x_norm_clstoken"]
                        patch_tokens = features["x_norm_patchtokens"]
                    elif "x" in features:
                        tokens = features["x"]
                        global_features = model.forward_head(tokens, pre_logits=True)
                        patch_tokens = tokens[:, 1:, :]
                    else:
                        raise ValueError("Unsupported timm DINOv2 feature dict.")
                else:
                    tokens = features
                    global_features = model.forward_head(tokens, pre_logits=True)
                    patch_tokens = tokens[:, 1:, :]

            if globals_cpu is None:
                n = len(df)
                globals_cpu = torch.empty((n, int(global_features.shape[-1])), dtype=storage_dtype)
                patches_cpu = torch.empty((n, int(patch_tokens.shape[1]), int(patch_tokens.shape[2])), dtype=storage_dtype)
            idx = indices.long()
            globals_cpu[idx] = global_features.detach().cpu().to(storage_dtype)
            patches_cpu[idx] = patch_tokens.detach().cpu().to(storage_dtype)

    payload = {
        "global_features": globals_cpu,
        "patch_tokens": patches_cpu,
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
        return cls(mean=x.mean(dim=0, keepdim=True), std=x.std(dim=0, keepdim=True).clamp_min(1e-6))

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
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainedClassifier:
    model: nn.Module
    scaler: FeatureScaler
    best_val_macro_f1: float
    best_epoch: int
    train_history: list[dict[str, float]]


def class_weights_from_labels(y: torch.Tensor, num_classes: int, device: torch.device, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = torch.bincount(y.cpu(), minlength=num_classes).float().clamp_min(1.0)
    raw = 1.0 / torch.sqrt(counts) if mode == "sqrt_inverse" else 1.0 / counts
    return (raw / raw.sum() * num_classes).to(device)


def build_classifier(model_type: str, in_dim: int, num_classes: int, args: argparse.Namespace) -> tuple[nn.Module, int]:
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
    criterion = nn.CrossEntropyLoss(weight=class_weights_from_labels(y_train, num_classes, device, args.class_weight_mode))
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
            loss = criterion(model(xb), yb)
            if l1 > 0:
                loss = loss + l1 * sum(param.abs().sum() for name, param in model.named_parameters() if "weight" in name)
            loss.backward()
            optimizer.step()
    return model


def train_classifier(
    name: str,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    class_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
    model_type: str,
    l1: float,
) -> TrainedClassifier:
    scaler = FeatureScaler.fit(x_train.float())
    x_train_s = scaler.transform(x_train.float())
    x_val_s = scaler.transform(x_val.float())
    model, epochs = build_classifier(model_type, x_train_s.shape[1], len(class_names), args)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_from_labels(y_train, len(class_names), device, args.class_weight_mode))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    y_train_device = y_train.to(device)
    y_val_np = y_val.cpu().numpy()
    best_state = copy.deepcopy(model.state_dict())
    best_score = -1.0
    best_epoch = 1
    patience = 0
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
            loss = criterion(model(xb), yb)
            if l1 > 0:
                loss = loss + l1 * sum(param.abs().sum() for name, param in model.named_parameters() if "weight" in name)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_idx)

        probs_val = predict_probs(model, scaler, x_val_s, device, already_scaled=True, move_to_cpu=False)
        val_metrics = classification_metrics(y_val_np, probs_val, class_names)
        val_macro = float(val_metrics["macro_f1_argmax"])
        history.append({"epoch": epoch + 1, "loss": total_loss / len(perm), "val_macro_f1": val_macro})
        if val_macro > best_score:
            best_score = val_macro
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                break

    model.load_state_dict(best_state)
    if args.refit_train_val:
        x_joint = torch.cat([x_train.float(), x_val.float()], dim=0)
        y_joint = torch.cat([y_train, y_val], dim=0)
        scaler = FeatureScaler.fit(x_joint)
        model, _ = build_classifier(model_type, x_joint.shape[1], len(class_names), args)
        model = fit_classifier_epochs(model, scaler.transform(x_joint), y_joint, len(class_names), args, device, best_epoch, l1)

    return TrainedClassifier(model=model.cpu(), scaler=scaler, best_val_macro_f1=best_score, best_epoch=best_epoch, train_history=history)


def predict_probs(
    model: nn.Module,
    scaler: FeatureScaler,
    x: torch.Tensor,
    device: torch.device,
    already_scaled: bool = False,
    batch_size: int = 4096,
    move_to_cpu: bool = True,
) -> np.ndarray:
    model = model.to(device).eval()
    x_scaled = x.float() if already_scaled else scaler.transform(x.float())
    probs = []
    with torch.inference_mode():
        for start in range(0, len(x_scaled), batch_size):
            logits = model(x_scaled[start : start + batch_size].to(device))
            probs.append(torch.softmax(logits, dim=1).detach().cpu())
    if move_to_cpu:
        model.cpu()
    return torch.cat(probs, dim=0).numpy()


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


def classification_metrics(y_true: np.ndarray, probs: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64)
    num_classes = len(class_names)
    y_pred = probs.argmax(axis=1)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred, strict=False):
        if 0 <= true < num_classes and 0 <= pred < num_classes:
            confusion[true, pred] += 1

    f1s = []
    recalls = []
    per_class_f1 = {}
    per_class_recall = {}
    for idx, name in enumerate(class_names):
        tp = confusion[idx, idx]
        fp = confusion[:, idx].sum() - tp
        fn = confusion[idx, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        f1s.append(f1)
        recalls.append(recall)
        per_class_f1[name] = float(f1)
        per_class_recall[name] = float(recall)

    onehot = np.eye(num_classes, dtype=bool)[y_true]
    aucs = [binary_auc(onehot[:, idx].astype(int), probs[:, idx]) for idx in range(num_classes)]
    aucs = [auc for auc in aucs if not math.isnan(auc)]
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1_argmax": float(np.mean(f1s)),
        "macro_auroc": float(np.mean(aucs)) if aucs else float("nan"),
        "balanced_accuracy": float(np.mean(recalls)),
        "per_class_f1": per_class_f1,
        "per_class_recall": per_class_recall,
    }


def select_indices(df: pd.DataFrame, split: str) -> np.ndarray:
    return df.index[df["split"].eq(split)].to_numpy(dtype=np.int64).copy()


def sample_train_patches(
    patch_tokens: torch.Tensor,
    train_idx: np.ndarray,
    max_patches: int,
    max_patches_per_image: int,
    seed: int,
) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    train_patches = patch_tokens[train_idx].float()
    n_img, n_patch, _ = train_patches.shape
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


def mini_batch_kmeans(x: torch.Tensor, k: int, n_iter: int, batch_size: int, spherical: bool, seed: int, device: torch.device) -> torch.Tensor:
    x = x.float().contiguous()
    n, dim = x.shape
    if n < k:
        raise ValueError(f"Need at least k samples for k-means, got n={n}, k={k}")
    gen = torch.Generator().manual_seed(seed)
    centroids = x[torch.randperm(n, generator=gen)[:k]].to(device)
    if spherical:
        centroids = normalize_rows(centroids)
    for _ in tqdm(range(n_iter), desc="k-means", leave=False):
        sums = torch.zeros((k, dim), dtype=torch.float32, device=device)
        counts = torch.zeros(k, dtype=torch.float32, device=device)
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, batch_size):
            batch = x[perm[start : start + batch_size]].to(device)
            if spherical:
                batch = normalize_rows(batch)
                labels = (batch @ centroids.T).argmax(dim=1)
            else:
                labels = torch.cdist(batch, centroids, p=2).pow(2).argmin(dim=1)
            sums.index_add_(0, labels, batch)
            counts += torch.bincount(labels, minlength=k).float()
        non_empty = counts > 0
        centroids[non_empty] = sums[non_empty] / counts[non_empty].unsqueeze(1)
        if spherical:
            centroids = normalize_rows(centroids)
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


def fit_euclidean_kmeans(sample: torch.Tensor, args: argparse.Namespace, device: torch.device) -> ConceptModel:
    mean = sample.mean(dim=0, keepdim=True)
    std = sample.std(dim=0, keepdim=True).clamp_min(1e-6)
    centroids = mini_batch_kmeans((sample - mean) / std, args.k, args.kmeans_iters, args.kmeans_batch_size, False, args.seed, device)
    return ConceptModel(method="euclidean_kmeans", centroids=centroids, mean=mean, std=std)


def fit_spherical_kmeans(sample: torch.Tensor, args: argparse.Namespace, device: torch.device) -> ConceptModel:
    centroids = mini_batch_kmeans(normalize_rows(sample), args.k, args.kmeans_iters, args.kmeans_batch_size, True, args.seed + 1, device)
    return ConceptModel(method="spherical_kmeans", centroids=centroids)


def fit_medoid_prototypes(sample: torch.Tensor, args: argparse.Namespace, device: torch.device) -> ConceptModel:
    x = normalize_rows(sample)
    centroids = mini_batch_kmeans(x, args.k, args.kmeans_iters, args.kmeans_batch_size, True, args.seed + 2, device)
    centroids_device = centroids.to(device)
    best_scores = torch.full((args.k,), -float("inf"), device=device)
    best_indices = torch.zeros(args.k, dtype=torch.long, device=device)
    for start in range(0, len(x), args.kmeans_batch_size):
        sims = x[start : start + args.kmeans_batch_size].to(device) @ centroids_device.T
        batch_scores, batch_indices = sims.max(dim=0)
        replace = batch_scores > best_scores
        best_scores[replace] = batch_scores[replace]
        best_indices[replace] = batch_indices[replace] + start
    return ConceptModel(method="medoid_prototype", centroids=x[best_indices.cpu()].contiguous())


def fit_gmm_pca(sample: torch.Tensor, args: argparse.Namespace, device: torch.device) -> ConceptModel:
    gen = torch.Generator().manual_seed(args.seed + 3)
    x = sample.float()
    x_for_pca = x[torch.randperm(len(x), generator=gen)[: args.pca_sample_size]] if len(x) > args.pca_sample_size else x
    pca_mean = x_for_pca.mean(dim=0, keepdim=True)
    centered = x_for_pca - pca_mean
    cov = centered.T @ centered / max(1, len(centered) - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    components = eigvecs[:, order[: min(args.pca_dim, eigvecs.shape[1])]].contiguous()
    z = (x - pca_mean) @ components
    init_means = mini_batch_kmeans(z, args.k, max(3, args.kmeans_iters // 2), args.kmeans_batch_size, False, args.seed + 4, device).to(device)
    means, vars_, weights = fit_diag_gmm(z, init_means, args.gmm_iters, args.kmeans_batch_size, device)
    return ConceptModel(method="gmm_pca", pca_mean=pca_mean, pca_components=components, gmm_means=means.cpu(), gmm_vars=vars_.cpu(), gmm_weights=weights.cpu())


def fit_diag_gmm(z: torch.Tensor, init_means: torch.Tensor, n_iter: int, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = z.float().contiguous()
    n, dim = z.shape
    k = init_means.shape[0]
    means = init_means.float().to(device)
    vars_ = z.var(dim=0, keepdim=True).clamp_min(1e-4).to(device).repeat(k, 1)
    weights = torch.full((k,), 1.0 / k, dtype=torch.float32, device=device)
    for _ in tqdm(range(n_iter), desc="diag GMM", leave=False):
        resp_sum = torch.zeros(k, dtype=torch.float32, device=device)
        x_sum = torch.zeros((k, dim), dtype=torch.float32, device=device)
        x2_sum = torch.zeros((k, dim), dtype=torch.float32, device=device)
        for start in range(0, n, batch_size):
            batch = z[start : start + batch_size].to(device)
            resp = torch.softmax(diag_gmm_log_prob(batch, means, vars_, weights), dim=1)
            resp_sum += resp.sum(dim=0)
            x_sum += resp.T @ batch
            x2_sum += resp.T @ (batch * batch)
        resp_safe = resp_sum.clamp_min(1e-6)
        means = x_sum / resp_safe.unsqueeze(1)
        vars_ = (x2_sum / resp_safe.unsqueeze(1) - means * means).clamp_min(1e-4)
        weights = resp_safe / resp_safe.sum()
    return means, vars_, weights


def diag_gmm_log_prob(x: torch.Tensor, means: torch.Tensor, vars_: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    x_exp = x.unsqueeze(1)
    mahal = ((x_exp - means.unsqueeze(0)).pow(2) / vars_.unsqueeze(0)).sum(dim=2)
    log_det = vars_.log().sum(dim=1).unsqueeze(0)
    return -0.5 * (mahal + log_det + x.shape[1] * math.log(2.0 * math.pi)) + weights.clamp_min(1e-8).log().unsqueeze(0)


def concept_activation_vectors(patch_tokens: torch.Tensor, concept_model: ConceptModel, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    n = patch_tokens.shape[0]
    activations = torch.empty((n, args.k), dtype=torch.float32)
    topk = min(args.topk_pool, patch_tokens.shape[1])

    with torch.inference_mode():
        for start in tqdm(range(0, n, args.activation_batch_size), desc=f"Activating {concept_model.method}", leave=False):
            batch = patch_tokens[start : start + args.activation_batch_size].float().to(device)
            b, p, d = batch.shape
            flat = batch.reshape(b * p, d)
            if concept_model.method == "euclidean_kmeans":
                flat = (flat - concept_model.mean.to(device)) / concept_model.std.to(device)
                scores = -torch.cdist(flat, concept_model.centroids.to(device), p=2).pow(2) / math.sqrt(flat.shape[1])
            elif concept_model.method in {"spherical_kmeans", "medoid_prototype"}:
                scores = normalize_rows(flat) @ normalize_rows(concept_model.centroids.to(device)).T
            else:
                z = (flat - concept_model.pca_mean.to(device)) @ concept_model.pca_components.to(device)
                scores = diag_gmm_log_prob(z, concept_model.gmm_means.to(device), concept_model.gmm_vars.to(device), concept_model.gmm_weights.to(device))
            activations[start : start + b] = scores.reshape(b, p, args.k).topk(topk, dim=1).values.mean(dim=1).cpu()
    return activations


def correlation_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = (a - a.mean(axis=0, keepdims=True)) / np.clip(a.std(axis=0, keepdims=True), 1e-8, None)
    b = (b - b.mean(axis=0, keepdims=True)) / np.clip(b.std(axis=0, keepdims=True), 1e-8, None)
    return (a.T @ b) / max(1, len(a) - 1)


def concept_quality(activations: torch.Tensor, skincon_concepts: np.ndarray, concept_names: list[str], concept_model: ConceptModel | None) -> dict[str, Any]:
    act_np = activations.numpy()
    corr = correlation_matrix(act_np, skincon_concepts)
    abs_corr = np.abs(corr)
    best_idx = abs_corr.argmax(axis=1)
    best_corr = abs_corr.max(axis=1)
    diversity = float("nan")
    if concept_model is not None and concept_model.centroids is not None:
        centroids = normalize_rows(concept_model.centroids.float())
        sims = torch.abs(centroids @ centroids.T).numpy()
        upper = sims[np.triu_indices_from(sims, k=1)]
        diversity = float(1.0 - upper.mean()) if len(upper) else float("nan")
    return {
        "skincon_alignment_mean_best_abs_corr": float(best_corr.mean()),
        "skincon_alignment_max_abs_corr": float(best_corr.max()),
        "diversity_proxy_1_minus_abs_centroid_cosine": diversity,
        "best_aligned_skincon_concepts": [
            {"concept": int(i), "skincon_concept": concept_names[int(best_idx[i])], "abs_corr": float(best_corr[i])}
            for i in range(len(best_idx))
        ],
    }


def top_concept_ablation(trained: TrainedClassifier, x_raw: torch.Tensor, device: torch.device, top_ms: tuple[int, ...] = (1, 3, 5)) -> dict[str, float]:
    if not isinstance(trained.model, LinearClassifier):
        return {}
    model = trained.model
    scaler = trained.scaler
    x_std = scaler.transform(x_raw.float())
    zero_std = scaler.transform(torch.zeros_like(x_raw[:1].float()))[0]
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
            top_idx = torch.argsort(weight[cls] * x_std[i], descending=True)[: min(m, x_std.shape[1])]
            ablated = x_std[i].clone()
            ablated[top_idx] = zero_std[top_idx]
            ablated_probs = torch.softmax(ablated @ weight.T + bias, dim=0)
            drops.append(float(probs[i, cls] - ablated_probs[cls]))
        out[f"drop_top_{m}_mean"] = float(np.mean(drops))
    return out


def run_experiment() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df, skincon_concepts, target_names = load_skincon_dataframe(args)
    df = stratified_image_split(df, args.seed)
    df.to_csv(output_dir / "skincon_manifest.csv", index=False)
    df.groupby(["split", "target_name"]).size().unstack(fill_value=0).to_csv(output_dir / "split_target_counts.csv")
    df[skincon_concepts].sum(axis=0).sort_values(ascending=False).rename("positive_count").to_csv(output_dir / "concept_positive_counts.csv")

    features = extract_or_load_features(args, df, output_dir, device)
    global_features = features["global_features"].float()
    patch_tokens = features["patch_tokens"]
    skincon_matrix = df[skincon_concepts].to_numpy(dtype=np.float32)
    y = torch.tensor(df["target_idx"].to_numpy(dtype=np.int64))
    y_np = y.numpy()
    train_idx = select_indices(df, "train")
    val_idx = select_indices(df, "val")
    test_idx = select_indices(df, "test")

    results: dict[str, Any] = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "device": str(device),
        "n_samples": int(len(df)),
        "target_names": target_names,
        "skincon_concepts": skincon_concepts,
        "split_counts": df.groupby("split").size().to_dict(),
        "experiments": {},
    }

    def train_and_eval(exp_id: str, name: str, x_all: torch.Tensor, model_type: str, l1: float, concept_model: ConceptModel | None = None) -> None:
        print(f"\nRunning {exp_id}: {name}")
        trained = train_classifier(name, x_all[train_idx], y[train_idx], x_all[val_idx], y[val_idx], target_names, args, device, model_type, l1)
        probs_all = predict_probs(trained.model, trained.scaler, x_all, device)
        test_metrics = classification_metrics(y_np[test_idx], probs_all[test_idx], target_names)
        val_metrics = classification_metrics(y_np[val_idx], probs_all[val_idx], target_names)
        result = {
            "name": name,
            "best_val_macro_f1": trained.best_val_macro_f1,
            "best_epoch": trained.best_epoch,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "top_concept_ablation": top_concept_ablation(trained, x_all[test_idx], device) if exp_id in {"E3", "E4", "E5", "E6", "E7"} else {},
            "concept_quality": concept_quality(x_all, skincon_matrix, skincon_concepts, concept_model) if exp_id in {"E4", "E5", "E6", "E7"} else {},
            "history_tail": trained.train_history[-10:],
        }
        results["experiments"][exp_id] = result
        save_json(results, output_dir / "results.json")

    if "E1" in args.models:
        train_and_eval("E1", "DINOv2 global linear probe", global_features, "linear", 0.0)
    if "E2" in args.models:
        train_and_eval("E2", "DINOv2 global MLP", global_features, "mlp", 0.0)
    if "E3" in args.models:
        train_and_eval("E3", "Oracle SkinCon concept sparse logistic", torch.tensor(skincon_matrix), "linear", args.l1)

    discovery_models = {
        "E4": ("Euclidean k-means discovered concepts", fit_euclidean_kmeans),
        "E5": ("Spherical k-means discovered concepts", fit_spherical_kmeans),
        "E6": ("GMM + PCA discovered concepts", fit_gmm_pca),
        "E7": ("Medoid prototype discovered concepts", fit_medoid_prototypes),
    }
    selected = [exp_id for exp_id in discovery_models if exp_id in args.models]
    if selected:
        patch_sample = sample_train_patches(patch_tokens, train_idx, args.max_patches_for_discovery, args.max_patches_per_image, args.seed)
        print(f"Discovery patch sample: {tuple(patch_sample.shape)}")
        for exp_id in selected:
            name, fitter = discovery_models[exp_id]
            print(f"\nFitting concepts for {exp_id}: {name}")
            concept_model = fitter(patch_sample, args, device)
            activations = concept_activation_vectors(patch_tokens, concept_model, args, device)
            torch.save({"activations": activations, "method": concept_model.method}, output_dir / f"activations_{exp_id}_{concept_model.method}_k{args.k}.pt")
            train_and_eval(exp_id, name, activations, "linear", args.l1, concept_model)

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
                "macro_auroc": metrics["macro_auroc"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "accuracy": metrics["accuracy"],
                "ablation_drop_top_1": ablation.get("drop_top_1_mean"),
                "ablation_drop_top_3": ablation.get("drop_top_3_mean"),
                "ablation_drop_top_5": ablation.get("drop_top_5_mean"),
                "skincon_alignment": quality.get("skincon_alignment_mean_best_abs_corr"),
                "diversity": quality.get("diversity_proxy_1_minus_abs_centroid_cosine"),
            }
        )
    summary = pd.DataFrame(rows).sort_values("id")
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)

    lines = [
        "# SkinCon E1-E7 Experiment Results",
        "",
        f"- Created at: {results['created_at']}",
        f"- Device: {results['device']}",
        f"- Samples: {results['n_samples']}",
        f"- Target classes: {len(results['target_names'])}",
        f"- SkinCon concepts: {len(results['skincon_concepts'])}",
        "",
        "## Test Metrics",
        "",
    ]
    lines.append(dataframe_to_markdown(summary) if not summary.empty else "_No completed experiments._")
    lines.extend(
        [
            "",
            "## Split Protocol",
            "",
            "- Split is image-level stratified by Fitzpatrick17k `label` after filtering classes with fewer than `min_target_count` samples.",
            "- Each diagnosis class is independently shuffled with the configured seed, then assigned approximately 70% train, 10% validation, and 20% test.",
            "- Classes with at least three samples are forced to contribute at least one validation and one test image. The default `min_target_count: 20` makes this safeguard non-binding for the main run.",
            "- No patient or lesion identifier is available in the SkinCon/Fitzpatrick17k metadata used here, so the split cannot guarantee patient-level or lesion-level de-duplication.",
            "- DINOv2 feature extraction runs on all images, but concept discovery for E4-E7 fits prototypes only from train split patch tokens. Validation is used for epoch selection, and test is used only for final reporting.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(header.replace("|", "\\|") for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.4f}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    run_experiment()
