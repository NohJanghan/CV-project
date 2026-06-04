import argparse
import copy
import csv
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


UCI_FEATURE_GROUPS = {
    "cap": ["cap-shape", "cap-surface", "cap-color", "bruises"],
    "gill": ["gill-attachment", "gill-spacing", "gill-size", "gill-color"],
    "stalk": [
        "stalk-shape",
        "stalk-root",
        "stalk-surface-above-ring",
        "stalk-surface-below-ring",
        "stalk-color-above-ring",
        "stalk-color-below-ring",
    ],
    "veil_ring": ["veil-type", "veil-color", "ring-number", "ring-type"],
    "spore": ["spore-print-color"],
    "ecology": ["population", "habitat"],
    "odor": ["odor"],
}

DEFAULT_VISIBLE_GROUPS = ["cap", "ecology"]


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

    parser = argparse.ArgumentParser(description="Mushroom image/species and UCI concept toxicity experiment.")
    parser.add_argument("--config_path", type=str, default=base_args.config_path)
    parser.add_argument(
        "--dataset1_zip",
        type=str,
        default=cfg.get("dataset1_zip", "data/mushrooms/raw/combined-kaggle-mushrooms-dataset.zip"),
    )
    parser.add_argument(
        "--dataset2_csv",
        type=str,
        default=cfg.get("dataset2_csv", "data/mushrooms/uci/mushrooms.csv"),
    )
    parser.add_argument(
        "--toxicity_csv",
        type=str,
        default=cfg.get("toxicity_csv", "data/mushrooms/species_toxicity_labels.csv"),
    )
    parser.add_argument("--output_dir", type=str, default=cfg.get("output_dir", "output/mushroom_toxicity"))
    parser.add_argument("--seed", type=int, default=cfg.get("seed", 42))
    parser.add_argument("--max_images_per_species", type=int, default=cfg.get("max_images_per_species", 60))
    parser.add_argument("--min_images_per_species", type=int, default=cfg.get("min_images_per_species", 20))
    parser.add_argument("--models", type=str, default=cfg.get("models", "M1,M2,M3,M4,M5"))

    parser.add_argument("--dinov2_model", type=str, default=cfg.get("dinov2_model", "vit_small_patch14_dinov2"))
    parser.add_argument("--pretrained", type=str2bool, default=cfg.get("pretrained", True))
    parser.add_argument("--image_size", type=int, default=cfg.get("image_size", 224))
    parser.add_argument("--feature_batch_size", type=int, default=cfg.get("feature_batch_size", 64))
    parser.add_argument("--feature_num_workers", type=int, default=cfg.get("feature_num_workers", 4))
    parser.add_argument("--feature_dtype", type=str, default=cfg.get("feature_dtype", "float16"))
    parser.add_argument("--amp", type=str2bool, default=cfg.get("amp", True))
    parser.add_argument("--force_features", type=str2bool, default=cfg.get("force_features", False))

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
    parser.add_argument(
        "--visible_concept_groups",
        type=str,
        default=cfg.get("visible_concept_groups", ",".join(DEFAULT_VISIBLE_GROUPS)),
        help="Comma-separated UCI concept groups visible from a typical top-view image.",
    )
    parser.add_argument("--unknown_train_prob", type=float, default=cfg.get("unknown_train_prob", 0.15))

    args = parser.parse_args()
    args.models = [m.strip().upper() for m in str(args.models).split(",") if m.strip()]
    args.visible_concept_groups = [
        group.strip() for group in str(args.visible_concept_groups).split(",") if group.strip()
    ]
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_manifest_from_zip(zip_path: str) -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.startswith("images/") or name.endswith("/"):
                continue
            parts = name.split("/")
            if len(parts) < 3:
                continue
            species = parts[1]
            lower = name.lower()
            if lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                rows.append({"species": species, "zip_path": name})
    return pd.DataFrame(rows)


def load_toxicity_labels(path: str) -> pd.DataFrame:
    labels = pd.read_csv(path)
    required = {"dataset_label", "binary_label_strict", "edibility_status"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"Toxicity CSV is missing required columns: {missing}")
    return labels


def build_image_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    manifest = make_manifest_from_zip(args.dataset1_zip)
    labels = load_toxicity_labels(args.toxicity_csv)
    df = manifest.merge(labels, left_on="species", right_on="dataset_label", how="left")
    df["binary_label_strict"] = df["binary_label_strict"].fillna("exclude")
    df = df.loc[df["binary_label_strict"].isin(["edible", "unsafe"])].copy()
    if df.empty:
        raise ValueError("No strict edible/unsafe species were available after applying toxicity labels.")
    df["target"] = df["binary_label_strict"].map({"edible": 0, "unsafe": 1}).astype(int)

    counts = df.groupby("species").size()
    valid_species = counts[counts >= args.min_images_per_species].index
    df = df.loc[df["species"].isin(valid_species)].copy()

    rng = np.random.default_rng(args.seed)
    pieces = []
    for species, part in df.groupby("species"):
        n = min(args.max_images_per_species, len(part))
        idx = rng.choice(part.index.to_numpy(), size=n, replace=False)
        pieces.append(df.loc[idx])
    out = pd.concat(pieces, axis=0).sort_values(["species", "zip_path"]).reset_index(drop=True)
    return stratified_species_split(out, args.seed)


def stratified_species_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    species_labels = df.groupby("species")["target"].first().reset_index()
    split_for_species: dict[str, str] = {}
    for target, part in species_labels.groupby("target"):
        species = part["species"].to_numpy()
        rng.shuffle(species)
        n = len(species)
        n_train = max(1, int(round(n * 0.70)))
        n_val = max(1, int(round(n * 0.10))) if n >= 3 else 0
        if n_train + n_val >= n:
            n_train = max(1, n - 2) if n >= 3 else max(1, n - 1)
            n_val = 1 if n >= 3 else 0
        for s in species[:n_train]:
            split_for_species[s] = "train"
        for s in species[n_train : n_train + n_val]:
            split_for_species[s] = "val"
        for s in species[n_train + n_val :]:
            split_for_species[s] = "test"
    out = df.copy()
    out["split"] = out["species"].map(split_for_species)
    return out


class MushroomZipDataset(Dataset):
    def __init__(self, df: pd.DataFrame, zip_path: str, image_size: int):
        self.df = df.reset_index(drop=True)
        self.zip_path = zip_path
        self._zip: zipfile.ZipFile | None = None
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def _zf(self) -> zipfile.ZipFile:
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.zip_path)
        return self._zip

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor]:
        zip_member = self.df.iloc[idx]["zip_path"]
        try:
            raw = self._zf().read(zip_member)
            image = Image.open(io.BytesIO(raw)).convert("RGB")
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
    return (
        output_dir
        / "features"
        / f"features_{args.dinov2_model}_img{args.image_size}_n{n_rows}_{args.feature_dtype}.pt"
    )


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

    dataset = MushroomZipDataset(df, args.dataset1_zip, args.image_size)
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
        for indices, images in tqdm(loader, desc="DINOv2 mushroom features", unit="batch"):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=autocast_enabled):
                tokens = model.forward_features(images)
                if isinstance(tokens, dict):
                    tokens = tokens.get("x_norm_patchtokens", tokens.get("x", None))
                    if tokens is None:
                        raise ValueError("Unsupported timm feature dict.")
                global_features = model.forward_head(tokens, pre_logits=True)
                patch_tokens = tokens[:, 1:, :]
            if globals_cpu is None:
                n = len(df)
                dim = int(global_features.shape[-1])
                n_patches = int(patch_tokens.shape[1])
                globals_cpu = torch.empty((n, dim), dtype=storage_dtype)
                patches_cpu = torch.empty((n, n_patches, dim), dtype=storage_dtype)
            idx = indices.long()
            globals_cpu[idx] = global_features.detach().cpu().to(storage_dtype)
            patches_cpu[idx] = patch_tokens.detach().cpu().to(storage_dtype)
    payload = {
        "global_features": globals_cpu,
        "patch_tokens": patches_cpu,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    torch.save(payload, cache_path)
    return payload


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
            logits = model(xb)
            loss = criterion(logits, yb)
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
    for _ in tqdm(range(n_iter), desc="Mushroom spherical k-means", leave=False):
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
        for start in tqdm(range(0, len(x), batch_size), desc="Mushroom medoid selection", leave=False):
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
        for start in tqdm(range(0, len(patches), args.activation_batch_size), desc="Mushroom concept activation", leave=False):
            batch = normalize_rows(patches[start : start + args.activation_batch_size].float().to(device).reshape(-1, patches.shape[-1]))
            b = min(args.activation_batch_size, len(patches) - start)
            scores = (batch @ centroids.T).reshape(b, patches.shape[1], centroids.shape[0])
            out[start : start + b] = scores.topk(topk, dim=1).values.mean(dim=1).cpu()
    return out


def build_uci_encoder(df: pd.DataFrame) -> dict[str, list[str]]:
    encoder = {}
    for col in df.columns:
        if col == "class":
            continue
        vals = sorted(str(v) for v in df[col].dropna().unique().tolist())
        if "?" not in vals:
            vals.append("?")
        if "__unknown__" not in vals:
            vals.append("__unknown__")
        encoder[col] = vals
    return encoder


def encode_uci(
    df: pd.DataFrame,
    encoder: dict[str, list[str]],
    unknown_cols: set[str] | None = None,
    rng: np.random.Generator | None = None,
    unknown_train_prob: float = 0.0,
) -> torch.Tensor:
    unknown_cols = unknown_cols or set()
    rows = []
    for _, row in df.iterrows():
        feats = []
        for col, vals in encoder.items():
            val = str(row[col])
            if col in unknown_cols or (rng is not None and rng.random() < unknown_train_prob):
                val = "__unknown__"
            vec = [0.0] * len(vals)
            if val in vals:
                vec[vals.index(val)] = 1.0
            elif "__unknown__" in vals:
                vec[vals.index("__unknown__")] = 1.0
            feats.extend(vec)
        rows.append(feats)
    return torch.tensor(rows, dtype=torch.float32)


def uci_unknown_columns(visible_groups: list[str]) -> set[str]:
    visible = set()
    for group in visible_groups:
        visible.update(UCI_FEATURE_GROUPS.get(group, []))
    all_cols = {col for cols in UCI_FEATURE_GROUPS.values() for col in cols}
    return all_cols - visible


def run_uci_concept_baseline(args: argparse.Namespace, output_dir: Path, device: torch.device) -> dict[str, Any]:
    df = pd.read_csv(args.dataset2_csv)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(df))
    train_end = int(len(df) * 0.70)
    val_end = int(len(df) * 0.85)
    train_df = df.iloc[perm[:train_end]].reset_index(drop=True)
    val_df = df.iloc[perm[train_end:val_end]].reset_index(drop=True)
    test_df = df.iloc[perm[val_end:]].reset_index(drop=True)
    encoder = build_uci_encoder(train_df)
    y_train = torch.tensor(train_df["class"].map({"e": 0, "p": 1}).to_numpy(dtype=np.int64))
    y_val = torch.tensor(val_df["class"].map({"e": 0, "p": 1}).to_numpy(dtype=np.int64))
    y_test = torch.tensor(test_df["class"].map({"e": 0, "p": 1}).to_numpy(dtype=np.int64))

    x_train = encode_uci(train_df, encoder, rng=rng, unknown_train_prob=args.unknown_train_prob)
    x_val = encode_uci(val_df, encoder)
    visible_unknown_cols = uci_unknown_columns(args.visible_concept_groups)
    x_test_full = encode_uci(test_df, encoder)
    x_test_visible = encode_uci(test_df, encoder, unknown_cols=visible_unknown_cols)
    trained = train_linear("M4 UCI concept unknown-aware", x_train, y_train, x_val, y_val, args, device, l1=args.l1)
    full_probs = predict_probs(trained.model, trained.scaler, x_test_full, device)
    visible_probs = predict_probs(trained.model, trained.scaler, x_test_visible, device)
    result = {
        "name": "M4 UCI tabular concept classifier with unknown support",
        "best_epoch": trained.best_epoch,
        "visible_groups": args.visible_concept_groups,
        "unknown_columns_at_top_view": sorted(visible_unknown_cols),
        "test_full_concepts": binary_metrics(y_test.numpy(), full_probs),
        "test_top_view_unknown": binary_metrics(y_test.numpy(), visible_probs),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
    }
    (output_dir / "uci_encoder.json").write_text(json.dumps(encoder, indent=2), encoding="utf-8")
    return result


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
    lines = [
        "# Mushroom Toxicity Experiment Report",
        "",
        f"- Created at: {results['created_at']}",
        f"- Device: {results['device']}",
        f"- Image samples: {results.get('n_image_samples', 0)}",
        f"- Strict labeled species: {results.get('n_strict_species', 0)}",
        "",
        "## Model Results",
        "",
        "| ID | Model | F1 unsafe | AUROC | Balanced accuracy | Recall unsafe | Notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for model_id, result in results["experiments"].items():
        if model_id == "M4":
            metrics = result["test_top_view_unknown"]
            notes = "UCI concepts; gill/stalk/spore marked unknown for top-view simulation"
        else:
            metrics = result["test_metrics"]
            notes = result.get("notes", "")
        lines.append(
            f"| {model_id} | {result['name']} | {metrics['f1']:.4f} | {metrics['auroc']:.4f} | "
            f"{metrics['balanced_accuracy']:.4f} | {metrics['recall_unsafe']:.4f} | {notes} |"
        )
    lines.extend(
        [
            "",
            "## Important Caveats",
            "",
            "- Dataset1 has image-species pairs, not image-toxicity labels.",
            "- Toxicity labels are generated by `download_mushroom_data.py` from external species-level weak-label sources with provenance.",
            "- Conditional, conflicting, missing, inedible, or uncertain species are excluded from strict metrics.",
            "- Dataset2 has concept-to-edible/poisonous labels but no species identity.",
            "- UCI is not used to label Dataset1 because its samples are hypothetical Agaricus/Lepiota morphology rows.",
            "- Therefore UCI concepts cannot directly supervise image concepts without additional image-concept annotations.",
            "- Some morphology concepts, especially gill, stalk, ring, and spore-print attributes, may be invisible in a single top-view image. M4 explicitly supports an `unknown` value for those concepts.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: dict[str, Any] = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "args": vars(args),
        "experiments": {},
    }

    if any(model_id in args.models for model_id in ["M1", "M2", "M3", "M5"]):
        image_df = build_image_dataframe(args)
        image_df.to_csv(output_dir / "dataset1_image_manifest_strict.csv", index=False)
        species_counts = image_df.groupby(["binary_label_strict", "species"]).size().reset_index(name="n")
        species_counts.to_csv(output_dir / "strict_species_counts.csv", index=False)
        results["n_image_samples"] = int(len(image_df))
        results["n_strict_species"] = int(image_df["species"].nunique())
        features = extract_or_load_features(args, image_df, output_dir, device)
        global_x = features["global_features"].float()
        patches = features["patch_tokens"]
        y = torch.tensor(image_df["target"].to_numpy(dtype=np.int64))
        train_idx = select_indices(image_df, "train")
        val_idx = select_indices(image_df, "val")
        test_idx = select_indices(image_df, "test")

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

        if "M1" in args.models:
            train_eval("M1", "DINOv2 global feature linear toxicity classifier", global_x, 0.0)

        if "M2" in args.models:
            pca_x = pca_reduce(global_x[train_idx], global_x, args.pca_dim)
            train_eval(
                "M2",
                f"DINOv2 global PCA-{args.pca_dim} linear control",
                pca_x,
                args.l1,
                "Dimension-reduction control, not concept-interpretable.",
            )

        if any(model_id in args.models for model_id in ["M3", "M5"]):
            sample = sample_train_patches(
                patches,
                train_idx,
                args.max_patches_for_discovery,
                args.max_patches_per_image,
                args.seed,
            )

        if "M3" in args.models:
            centroids = mini_batch_spherical_kmeans(
                sample,
                args.k,
                args.kmeans_iters,
                args.kmeans_batch_size,
                args.seed,
                device,
            )
            activations = concept_activations(patches, centroids, args, device)
            torch.save({"activations": activations, "centroids": centroids}, output_dir / f"mushroom_concepts_k{args.k}.pt")
            train_eval(
                "M3",
                f"DINOv2 patch spherical concept bottleneck K={args.k}",
                activations,
                args.l1,
                "Latent visual concepts; requires top-activating image review for naming.",
            )

        if "M5" in args.models:
            medoids = fit_medoid_prototypes(
                sample,
                args.k,
                args.kmeans_iters,
                args.kmeans_batch_size,
                args.seed + 2,
                device,
            )
            activations = concept_activations(patches, medoids, args, device)
            torch.save(
                {"activations": activations, "medoids": medoids},
                output_dir / f"mushroom_medoids_k{args.k}.pt",
            )
            train_eval(
                "M5",
                f"MILK10K E7-style medoid prototype concept bottleneck K={args.k}",
                activations,
                args.l1,
                "Actual train-patch prototypes; easier to inspect than free centroids.",
            )

    if "M4" in args.models:
        results["experiments"]["M4"] = run_uci_concept_baseline(args, output_dir, device)
        save_json(results, output_dir / "results.json")

    save_json(results, output_dir / "results.json")
    write_report(results, output_dir)
    print(f"Done. Results written to {output_dir}")


if __name__ == "__main__":
    run()
