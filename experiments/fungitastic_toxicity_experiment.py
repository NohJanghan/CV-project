import argparse
import copy
import datetime as dt
import io
import json
import math
import random
import zipfile
from dataclasses import dataclass
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
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config_path", type=str, default=None)
    base_args, _ = base_parser.parse_known_args()
    cfg = load_yaml(base_args.config_path)

    parser = argparse.ArgumentParser(description="Aligned FungiTastic toxicity concept experiment.")
    parser.add_argument("--config_path", type=str, default=base_args.config_path)
    parser.add_argument("--metadata_dir", type=str, default=cfg.get("metadata_dir", "data/fungitastic/metadata/FungiTastic-Mini"))
    parser.add_argument("--image_root", type=str, default=cfg.get("image_root", "data/fungitastic"))
    parser.add_argument("--output_dir", type=str, default=cfg.get("output_dir", "output/fungitastic_toxicity"))
    parser.add_argument("--dataset_variant", type=str, default=cfg.get("dataset_variant", "mini"), choices=["mini", "full"])
    parser.add_argument(
        "--evaluation_protocol",
        type=str,
        default=cfg.get("evaluation_protocol", "openset"),
        choices=["openset", "closedset"],
        help="For full FungiTastic, choose either OpenSet or ClosedSet val/test metadata. Ignored for Mini.",
    )
    parser.add_argument("--image_size", type=int, default=cfg.get("image_size", 224))
    parser.add_argument("--fungitastic_resolution", type=str, default=cfg.get("fungitastic_resolution", "300p"))
    parser.add_argument("--seed", type=int, default=cfg.get("seed", 42))
    parser.add_argument("--models", type=str, default=cfg.get("models", "F1,F2,F3,F4"))

    parser.add_argument("--max_images_per_species_per_split", type=int, default=cfg.get("max_images_per_species_per_split", 60))
    parser.add_argument("--negative_to_positive_ratio", type=float, default=cfg.get("negative_to_positive_ratio", 2.0))
    parser.add_argument("--min_images_per_species", type=int, default=cfg.get("min_images_per_species", 5))
    parser.add_argument("--require_existing_images", type=str2bool, default=cfg.get("require_existing_images", True))

    parser.add_argument("--dinov2_model", type=str, default=cfg.get("dinov2_model", "vit_small_patch14_dinov2"))
    parser.add_argument("--pretrained", type=str2bool, default=cfg.get("pretrained", True))
    parser.add_argument("--feature_batch_size", type=int, default=cfg.get("feature_batch_size", 64))
    parser.add_argument("--feature_num_workers", type=int, default=cfg.get("feature_num_workers", 4))
    parser.add_argument("--feature_dtype", type=str, default=cfg.get("feature_dtype", "float16"))
    parser.add_argument("--amp", type=str2bool, default=cfg.get("amp", True))
    parser.add_argument("--force_features", type=str2bool, default=cfg.get("force_features", False))
    parser.add_argument("--store_patch_tokens", type=str2bool, default=cfg.get("store_patch_tokens", True))

    parser.add_argument("--k", type=int, default=cfg.get("k", 40))
    parser.add_argument("--topk_pool", type=int, default=cfg.get("topk_pool", 5))
    parser.add_argument("--max_patches_for_discovery", type=int, default=cfg.get("max_patches_for_discovery", 120000))
    parser.add_argument("--max_patches_per_image", type=int, default=cfg.get("max_patches_per_image", 48))
    parser.add_argument("--kmeans_iters", type=int, default=cfg.get("kmeans_iters", 20))
    parser.add_argument("--kmeans_batch_size", type=int, default=cfg.get("kmeans_batch_size", 8192))
    parser.add_argument("--activation_batch_size", type=int, default=cfg.get("activation_batch_size", 128))
    parser.add_argument("--pca_dim", type=int, default=cfg.get("pca_dim", 64))

    parser.add_argument("--epochs", type=int, default=cfg.get("epochs", 120))
    parser.add_argument("--train_batch_size", type=int, default=cfg.get("train_batch_size", 256))
    parser.add_argument("--lr", type=float, default=cfg.get("lr", 1e-3))
    parser.add_argument("--weight_decay", type=float, default=cfg.get("weight_decay", 1e-4))
    parser.add_argument("--l1", type=float, default=cfg.get("l1", 1e-5))
    parser.add_argument("--early_stop_patience", type=int, default=cfg.get("early_stop_patience", 25))
    parser.add_argument("--refit_train_val", type=str2bool, default=cfg.get("refit_train_val", True))

    args = parser.parse_args()
    args.models = [m.strip().upper() for m in str(args.models).split(",") if m.strip()]
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_csv_paths(metadata_dir: Path, variant: str, evaluation_protocol: str = "openset") -> list[tuple[str, Path]]:
    if variant == "full":
        protocol_files = {
            "openset": ("OpenSet", "OpenSet"),
            "closedset": ("ClosedSet", "ClosedSet"),
        }
        if evaluation_protocol not in protocol_files:
            raise ValueError(f"Unknown FungiTastic evaluation protocol: {evaluation_protocol}")
        val_name, test_name = protocol_files[evaluation_protocol]
        return [
            ("train", metadata_dir / "FungiTastic-Train.csv"),
            ("val", metadata_dir / f"FungiTastic-{val_name}-Val.csv"),
            ("test", metadata_dir / f"FungiTastic-{test_name}-Test.csv"),
        ]
    return [
        ("train", metadata_dir / "FungiTastic-Mini-Train.csv"),
        ("val", metadata_dir / "FungiTastic-Mini-Val.csv"),
        ("test", metadata_dir / "FungiTastic-Mini-Test.csv"),
    ]


def load_fungitastic_manifest(args: argparse.Namespace) -> pd.DataFrame:
    metadata_dir = Path(args.metadata_dir)
    rows = []
    keep_cols = {
        "eventDate",
        "year",
        "month",
        "habitat",
        "scientificName",
        "family",
        "genus",
        "species",
        "substrate",
        "filename",
        "category_id",
        "metaSubstrate",
        "poisonous",
    }
    for split, path in split_csv_paths(metadata_dir, args.dataset_variant, args.evaluation_protocol):
        df = pd.read_csv(path, usecols=lambda col: col in keep_cols)
        df["split"] = split
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    df = df.loc[df["poisonous"].isin([0, 1])].copy()
    df["target"] = df["poisonous"].astype(int)
    df["image_path"] = df.apply(lambda row: image_path(args, row["split"], row["filename"]), axis=1)

    counts = df.groupby("species").size()
    valid_species = counts[counts >= args.min_images_per_species].index
    df = df.loc[df["species"].isin(valid_species)].copy()

    if args.require_existing_images:
        exists = df["image_path"].map(lambda p: Path(p).exists())
        missing = int((~exists).sum())
        if missing:
            raise FileNotFoundError(
                f"{missing} FungiTastic image files are missing. Run download_fungitastic.py first."
            )
        df = df.loc[exists].copy()

    return balanced_sample(df, args)


def image_path(args: argparse.Namespace, split: str, filename: str) -> str:
    subset_dir = "FungiTastic" if args.dataset_variant == "full" else "FungiTastic-Mini"
    return str(Path(args.image_root) / subset_dir / split / args.fungitastic_resolution / str(filename))


def balanced_sample(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    capped = []
    for (split, species), part in df.groupby(["split", "species"]):
        n = min(args.max_images_per_species_per_split, len(part))
        idx = rng.choice(part.index.to_numpy(), size=n, replace=False)
        capped.append(df.loc[idx])
    df = pd.concat(capped, axis=0)

    balanced = []
    for split, part in df.groupby("split"):
        positives = part.loc[part["target"].eq(1)]
        negatives = part.loc[part["target"].eq(0)]
        if args.negative_to_positive_ratio and args.negative_to_positive_ratio > 0:
            max_neg = int(round(len(positives) * args.negative_to_positive_ratio))
            if max_neg > 0 and len(negatives) > max_neg:
                neg_idx = rng.choice(negatives.index.to_numpy(), size=max_neg, replace=False)
                negatives = part.loc[neg_idx]
        balanced.append(pd.concat([positives, negatives], axis=0))
    out = pd.concat(balanced, axis=0).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    return out


class FungiTasticImageDataset(Dataset):
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
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))
        return idx, self.transform(image)


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


def feature_cache_path(args: argparse.Namespace, output_dir: Path, n_rows: int) -> Path:
    suffix = "withpatches" if args.store_patch_tokens else "globalonly"
    return output_dir / "features" / f"features_{args.dinov2_model}_img{args.image_size}_n{n_rows}_{args.feature_dtype}_{suffix}.pt"


def extract_or_load_features(
    args: argparse.Namespace,
    df: pd.DataFrame,
    output_dir: Path,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    cache_path = feature_cache_path(args, output_dir, len(df))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not args.force_features:
        print(f"Loading cached features: {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    dataset = FungiTasticImageDataset(df, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.feature_batch_size,
        shuffle=False,
        num_workers=args.feature_num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_dinov2(args, device)
    storage_dtype = torch.float16 if args.feature_dtype == "float16" else torch.float32
    globals_cpu = None
    patches_cpu = None
    autocast_enabled = bool(args.amp and device.type == "cuda")
    with torch.inference_mode():
        for indices, images in tqdm(loader, desc="DINOv2 FungiTastic features", unit="batch"):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=autocast_enabled):
                tokens = model.forward_features(images)
                if isinstance(tokens, dict):
                    tokens = tokens.get("x_norm_patchtokens", tokens.get("x", None))
                    if tokens is None:
                        raise ValueError("Unsupported timm feature dict.")
                global_features = model.forward_head(tokens, pre_logits=True)
                patch_tokens = tokens[:, 1:, :] if args.store_patch_tokens else None
            if globals_cpu is None:
                n = len(df)
                dim = int(global_features.shape[-1])
                globals_cpu = torch.empty((n, dim), dtype=storage_dtype)
                if args.store_patch_tokens:
                    n_patches = int(patch_tokens.shape[1])
                    patches_cpu = torch.empty((n, n_patches, dim), dtype=storage_dtype)
            idx = indices.long()
            globals_cpu[idx] = global_features.detach().cpu().to(storage_dtype)
            if args.store_patch_tokens:
                patches_cpu[idx] = patch_tokens.detach().cpu().to(storage_dtype)
    payload = {
        "global_features": globals_cpu,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    if args.store_patch_tokens:
        payload["patch_tokens"] = patches_cpu
    torch.save(payload, cache_path)
    return payload


def sampled_patches_cache_path(args: argparse.Namespace, output_dir: Path) -> Path:
    return (
        output_dir
        / "features"
        / f"sampled_train_patches_{args.dinov2_model}_img{args.image_size}_max{args.max_patches_for_discovery}_{args.feature_dtype}.pt"
    )


def sample_train_patches_stream(
    args: argparse.Namespace,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    output_dir: Path,
    device: torch.device,
) -> torch.Tensor:
    cache_path = sampled_patches_cache_path(args, output_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not args.force_features:
        print(f"Loading cached sampled patches: {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=False)["patches"].float()

    dataset = FungiTasticImageDataset(df.iloc[train_idx].reset_index(drop=True), args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.feature_batch_size,
        shuffle=False,
        num_workers=args.feature_num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_dinov2(args, device)
    gen = torch.Generator().manual_seed(args.seed)
    chunks = []
    total = 0
    autocast_enabled = bool(args.amp and device.type == "cuda")
    with torch.inference_mode():
        for _, images in tqdm(loader, desc="Streaming train patch sample", unit="batch"):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=autocast_enabled):
                tokens = model.forward_features(images)
                if isinstance(tokens, dict):
                    tokens = tokens.get("x_norm_patchtokens", tokens.get("x", None))
                    if tokens is None:
                        raise ValueError("Unsupported timm feature dict.")
                patch_tokens = tokens[:, 1:, :].detach().cpu().float()
            n_img, n_patch, _ = patch_tokens.shape
            per = min(args.max_patches_per_image, n_patch)
            for i in range(n_img):
                idx = torch.randperm(n_patch, generator=gen)[:per]
                chunks.append(patch_tokens[i, idx])
                total += per
                if total >= args.max_patches_for_discovery:
                    break
            if total >= args.max_patches_for_discovery:
                break
    patches = torch.cat(chunks, dim=0)[: args.max_patches_for_discovery].contiguous()
    torch.save({"patches": patches, "created_at": dt.datetime.now().isoformat(timespec="seconds")}, cache_path)
    return patches


def concept_activations_stream(
    args: argparse.Namespace,
    df: pd.DataFrame,
    centroids: torch.Tensor,
    output_dir: Path,
    device: torch.device,
    cache_name: str,
) -> torch.Tensor:
    cache_path = output_dir / "features" / f"{cache_name}_activations_k{centroids.shape[0]}.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not args.force_features:
        print(f"Loading cached streamed activations: {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=False)["activations"].float()

    dataset = FungiTasticImageDataset(df, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.feature_batch_size,
        shuffle=False,
        num_workers=args.feature_num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_dinov2(args, device)
    centroids = normalize_rows(centroids.float()).to(device)
    out = torch.empty((len(df), centroids.shape[0]), dtype=torch.float32)
    topk = min(args.topk_pool, 256)
    autocast_enabled = bool(args.amp and device.type == "cuda")
    with torch.inference_mode():
        for indices, images in tqdm(loader, desc=f"Streaming {cache_name} activations", unit="batch"):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=autocast_enabled):
                tokens = model.forward_features(images)
                if isinstance(tokens, dict):
                    tokens = tokens.get("x_norm_patchtokens", tokens.get("x", None))
                    if tokens is None:
                        raise ValueError("Unsupported timm feature dict.")
                patches = tokens[:, 1:, :]
            b, p, d = patches.shape
            flat = normalize_rows(patches.float().reshape(b * p, d))
            scores = (flat @ centroids.T).reshape(b, p, centroids.shape[0])
            pooled = scores.topk(min(topk, p), dim=1).values.mean(dim=1)
            out[indices.long()] = pooled.detach().cpu()
    torch.save({"activations": out, "created_at": dt.datetime.now().isoformat(timespec="seconds")}, cache_path)
    return out


@dataclass
class FeatureScaler:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, x: torch.Tensor) -> "FeatureScaler":
        return cls(x.mean(dim=0, keepdim=True), x.std(dim=0, keepdim=True).clamp_min(1e-6))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class LinearClassifier(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@dataclass
class TrainedClassifier:
    model: LinearClassifier
    scaler: FeatureScaler
    best_epoch: int
    best_val_f1: float
    history: list[dict[str, float]]


def class_weights(y: torch.Tensor, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(y.cpu(), minlength=2).float().clamp_min(1.0)
    raw = 1.0 / torch.sqrt(counts)
    return (raw / raw.sum() * 2).to(device)


def train_linear(
    name: str,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    l1: float = 0.0,
) -> TrainedClassifier:
    scaler = FeatureScaler.fit(x_train.float())
    x_train_s = scaler.transform(x_train.float())
    x_val_s = scaler.transform(x_val.float())
    model = LinearClassifier(x_train_s.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(y_train, device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    y_train_device = y_train.to(device)
    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 1
    patience = 0
    history = []
    for epoch in tqdm(range(args.epochs), desc=f"Training {name}", leave=False):
        model.train()
        perm = torch.randperm(len(x_train_s))
        for start in range(0, len(perm), args.train_batch_size):
            batch_idx = perm[start : start + args.train_batch_size]
            xb = x_train_s[batch_idx].to(device)
            yb = y_train_device[batch_idx]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            if l1 > 0:
                loss = loss + l1 * model.linear.weight.abs().sum()
            loss.backward()
            optimizer.step()
        probs = predict_probs(model, scaler, x_val, device, move_to_cpu=False)
        metrics = binary_metrics(y_val.numpy(), probs)
        history.append({"epoch": epoch + 1, "val_f1": metrics["f1"], "val_balanced_accuracy": metrics["balanced_accuracy"]})
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                break
    model.load_state_dict(best_state)
    if args.refit_train_val:
        refit_x = torch.cat([x_train.float(), x_val.float()], dim=0)
        refit_y = torch.cat([y_train, y_val], dim=0)
        scaler = FeatureScaler.fit(refit_x)
        refit_x_s = scaler.transform(refit_x)
        model = LinearClassifier(refit_x_s.shape[1]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights(refit_y, device))
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        refit_y_device = refit_y.to(device)
        for _ in range(best_epoch):
            model.train()
            perm = torch.randperm(len(refit_x_s))
            for start in range(0, len(perm), args.train_batch_size):
                batch_idx = perm[start : start + args.train_batch_size]
                xb = refit_x_s[batch_idx].to(device)
                yb = refit_y_device[batch_idx]
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
                if l1 > 0:
                    loss = loss + l1 * model.linear.weight.abs().sum()
                loss.backward()
                optimizer.step()
    return TrainedClassifier(model.cpu(), scaler, best_epoch, best_f1, history)


def predict_probs(
    model: nn.Module,
    scaler: FeatureScaler,
    x: torch.Tensor,
    device: torch.device,
    batch_size: int = 4096,
    move_to_cpu: bool = True,
) -> np.ndarray:
    model.to(device).eval()
    x_s = scaler.transform(x.float())
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(x_s), batch_size):
            logits = model(x_s[start : start + batch_size].to(device))
            chunks.append(torch.softmax(logits, dim=1).detach().cpu())
    if move_to_cpu:
        model.cpu()
    return torch.cat(chunks, dim=0).numpy()


def binary_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy()
    rank_sum = ranks[y == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binary_metrics(y: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    pred = probs.argmax(axis=1)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "accuracy": float((pred == y).mean()),
        "precision_unsafe": float(precision),
        "recall_unsafe": float(recall),
        "specificity_edible": float(specificity),
        "balanced_accuracy": float((recall + specificity) / 2),
        "f1": float(f1),
        "auroc": binary_auc(y, probs[:, 1]),
        "n": int(len(y)),
    }


def select_indices(df: pd.DataFrame, split: str) -> np.ndarray:
    return df.index[df["split"].eq(split)].to_numpy(dtype=np.int64).copy()


def pca_reduce(train_x: torch.Tensor, all_x: torch.Tensor, dim: int) -> torch.Tensor:
    scaler = FeatureScaler.fit(train_x.float())
    train_s = scaler.transform(train_x.float())
    all_s = scaler.transform(all_x.float())
    cov = train_s.T @ train_s / max(1, len(train_s) - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)[:dim]
    components = eigvecs[:, order].contiguous()
    return all_s @ components


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=1, eps=1e-6)


def sample_train_patches(
    patches: torch.Tensor,
    train_idx: np.ndarray,
    max_patches: int,
    max_per_image: int,
    seed: int,
) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    x = patches[train_idx].float()
    n_img, n_patch, _ = x.shape
    per = min(max_per_image, n_patch)
    chunks = []
    for i in range(n_img):
        idx = torch.randperm(n_patch, generator=gen)[:per]
        chunks.append(x[i, idx])
    sample = torch.cat(chunks, dim=0)
    if len(sample) > max_patches:
        idx = torch.randperm(len(sample), generator=gen)[:max_patches]
        sample = sample[idx]
    return sample


def mini_batch_spherical_kmeans(
    x: torch.Tensor,
    k: int,
    n_iter: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    x = normalize_rows(x.float().contiguous())
    n, dim = x.shape
    gen = torch.Generator().manual_seed(seed)
    centroids = x[torch.randperm(n, generator=gen)[:k]].to(device)
    centroids = normalize_rows(centroids)
    for _ in tqdm(range(n_iter), desc="FungiTastic spherical k-means", leave=False):
        sums = torch.zeros((k, dim), device=device)
        counts = torch.zeros(k, device=device)
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, batch_size):
            batch = x[perm[start : start + batch_size]].to(device)
            labels = (batch @ centroids.T).argmax(dim=1)
            sums.index_add_(0, labels, batch)
            counts += torch.bincount(labels, minlength=k).float()
        non_empty = counts > 0
        centroids[non_empty] = sums[non_empty] / counts[non_empty].unsqueeze(1)
        centroids = normalize_rows(centroids)
    return centroids.cpu()


def fit_medoid_prototypes(
    sample: torch.Tensor,
    k: int,
    n_iter: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    x = normalize_rows(sample.float().contiguous())
    centroids = mini_batch_spherical_kmeans(x, k, n_iter, batch_size, seed, device)
    centroids_device = normalize_rows(centroids.float()).to(device)
    best_scores = torch.full((k,), -float("inf"), device=device)
    best_indices = torch.zeros(k, dtype=torch.long, device=device)
    with torch.inference_mode():
        for start in tqdm(range(0, len(x), batch_size), desc="FungiTastic medoid selection", leave=False):
            batch = x[start : start + batch_size].to(device)
            sims = batch @ centroids_device.T
            batch_scores, batch_indices = sims.max(dim=0)
            replace = batch_scores > best_scores
            best_scores[replace] = batch_scores[replace]
            best_indices[replace] = batch_indices[replace] + start
    return x[best_indices.cpu()].contiguous()


def concept_activations(
    patches: torch.Tensor,
    centroids: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    centroids = normalize_rows(centroids.float()).to(device)
    out = torch.empty((len(patches), centroids.shape[0]), dtype=torch.float32)
    topk = min(args.topk_pool, patches.shape[1])
    with torch.inference_mode():
        for start in tqdm(range(0, len(patches), args.activation_batch_size), desc="FungiTastic concept activation", leave=False):
            batch = normalize_rows(patches[start : start + args.activation_batch_size].float().to(device).reshape(-1, patches.shape[-1]))
            b = min(args.activation_batch_size, len(patches) - start)
            scores = (batch @ centroids.T).reshape(b, patches.shape[1], centroids.shape[0])
            out[start : start + b] = scores.topk(topk, dim=1).values.mean(dim=1).cpu()
    return out


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
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_report(results: dict[str, Any], output_dir: Path) -> None:
    negative_ratio = results.get("args", {}).get("negative_to_positive_ratio")
    if negative_ratio and negative_ratio > 0:
        sampling_note = (
            f"- Non-poisonous rows are sampled up to {negative_ratio:g}x poisonous rows per split "
            "while retaining official train/val/test splits."
        )
    else:
        sampling_note = "- Non-poisonous rows are not downsampled; all eligible metadata rows with existing images are used."
    lines = [
        "# FungiTastic Toxicity Experiment Report",
        "",
        f"- Created at: {results['created_at']}",
        f"- Device: {results['device']}",
        f"- Image samples: {results.get('n_image_samples', 0)}",
        f"- Species: {results.get('n_species', 0)}",
        f"- Dataset variant: {results.get('dataset_variant', 'unknown')}",
        f"- Evaluation protocol: {results.get('evaluation_protocol', 'openset')}",
        "- Label source: FungiTastic metadata `poisonous` column, not Wikipedia or UCI.",
        "",
        "## Model Results",
        "",
        "| ID | Model | F1 unsafe | AUROC | Balanced accuracy | Recall unsafe | Notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for model_id, result in results["experiments"].items():
        metrics = result["test_metrics"]
        lines.append(
            f"| {model_id} | {result['name']} | {metrics['f1']:.4f} | {metrics['auroc']:.4f} | "
            f"{metrics['balanced_accuracy']:.4f} | {metrics['recall_unsafe']:.4f} | {result.get('notes', '')} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- FungiTastic aligns image rows, species metadata, and poisonous labels in one dataset.",
            "- UCI Mushroom Classification is not used because its species set and hypothetical tabular samples do not align with the image dataset.",
            sampling_note,
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_fungitastic_manifest(args)
    df.to_csv(output_dir / "manifest.csv", index=False)
    split_counts = df.groupby(["split", "target"]).size().reset_index(name="n")
    split_counts.to_csv(output_dir / "split_counts.csv", index=False)
    species_counts = df.groupby(["target", "species"]).size().reset_index(name="n")
    species_counts.to_csv(output_dir / "species_counts.csv", index=False)

    results: dict[str, Any] = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "args": vars(args),
        "dataset_variant": args.dataset_variant,
        "evaluation_protocol": args.evaluation_protocol,
        "n_image_samples": int(len(df)),
        "n_species": int(df["species"].nunique()),
        "experiments": {},
        "split_counts": split_counts.to_dict(orient="records"),
    }

    features = extract_or_load_features(args, df, output_dir, device)
    global_x = features["global_features"].float()
    patches = features.get("patch_tokens")
    y = torch.tensor(df["target"].to_numpy(dtype=np.int64))
    train_idx = select_indices(df, "train")
    val_idx = select_indices(df, "val")
    test_idx = select_indices(df, "test")

    def train_eval(model_id: str, name: str, x: torch.Tensor, l1: float, notes: str = "") -> None:
        trained = train_linear(name, x[train_idx], y[train_idx], x[val_idx], y[val_idx], args, device, l1=l1)
        probs = predict_probs(trained.model, trained.scaler, x[test_idx], device)
        metrics = binary_metrics(y[test_idx].numpy(), probs)
        results["experiments"][model_id] = {
            "name": name,
            "best_epoch": trained.best_epoch,
            "best_val_f1": trained.best_val_f1,
            "test_metrics": metrics,
            "notes": notes,
        }
        save_json(results, output_dir / "results.json")

    if "F1" in args.models:
        train_eval("F1", "DINOv2 global feature linear toxicity classifier", global_x, 0.0)

    if "F2" in args.models:
        pca_x = pca_reduce(global_x[train_idx], global_x, args.pca_dim)
        train_eval(
            "F2",
            f"DINOv2 global PCA-{args.pca_dim} linear control",
            pca_x,
            args.l1,
            "Dimension-reduction control, not concept-interpretable.",
        )

    if any(model_id in args.models for model_id in ["F3", "F4"]):
        if patches is None:
            sample = sample_train_patches_stream(args, df, train_idx, output_dir, device)
        else:
            sample = sample_train_patches(
                patches,
                train_idx,
                args.max_patches_for_discovery,
                args.max_patches_per_image,
                args.seed,
            )

    if "F3" in args.models:
        centroids = mini_batch_spherical_kmeans(
            sample,
            args.k,
            args.kmeans_iters,
            args.kmeans_batch_size,
            args.seed,
            device,
        )
        if patches is None:
            activations = concept_activations_stream(args, df, centroids, output_dir, device, "fungitastic_concepts")
        else:
            activations = concept_activations(patches, centroids, args, device)
        torch.save({"activations": activations, "centroids": centroids}, output_dir / f"fungitastic_concepts_k{args.k}.pt")
        train_eval(
            "F3",
            f"DINOv2 patch spherical concept bottleneck K={args.k}",
            activations,
            args.l1,
            "Latent visual concepts; requires top-activating image review for naming.",
        )

    if "F4" in args.models:
        medoids = fit_medoid_prototypes(
            sample,
            args.k,
            args.kmeans_iters,
            args.kmeans_batch_size,
            args.seed + 2,
            device,
        )
        if patches is None:
            activations = concept_activations_stream(args, df, medoids, output_dir, device, "fungitastic_medoids")
        else:
            activations = concept_activations(patches, medoids, args, device)
        torch.save({"activations": activations, "medoids": medoids}, output_dir / f"fungitastic_medoids_k{args.k}.pt")
        train_eval(
            "F4",
            f"MILK10K E7-style medoid prototype concept bottleneck K={args.k}",
            activations,
            args.l1,
            "Actual train-patch prototypes; easier to inspect than free centroids.",
        )

    save_json(results, output_dir / "results.json")
    write_report(results, output_dir)
    print(f"Done. Results written to {output_dir}")


if __name__ == "__main__":
    run()
