"""
Download only the FungiTastic pieces needed for this project's toxicity experiment.

The full Kaggle archive is about 50GB. By default this script downloads:

    - metadata.zip
    - FungiTastic-Mini train/val/test images at 300px by default

That is enough to run an aligned image -> poisonous experiment without relying on
Wikipedia labels or the disconnected UCI mushroom dataset.

Usage:
    uv run python download_fungitastic.py
    uv run python download_fungitastic.py --metadata-only
    uv run python download_fungitastic.py --include-dna-test
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


DOWNLOAD_ROOT = "https://cmp.felk.cvut.cz/datagrid/FungiTastic/shared/download"
USER_AGENT = "cv-project-fungitastic-downloader/1.0"


@dataclass(frozen=True)
class Asset:
    key: str
    filename: str
    target_dir: Path
    description: str

    @property
    def url(self) -> str:
        return f"{DOWNLOAD_ROOT}/{self.filename}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a small FungiTastic subset for toxicity experiments.")
    parser.add_argument("--output-dir", type=Path, default=Path("data") / "fungitastic")
    parser.add_argument("--subset", choices=["m", "full"], default="m", help="Subset to download. Default: m.")
    parser.add_argument("--size", choices=["300", "500", "720", "fullsize"], default="300")
    parser.add_argument("--metadata-only", action="store_true", help="Download/extract metadata but skip images.")
    parser.add_argument("--include-dna-test", action="store_true", help="Also download the Mini dna-test split.")
    parser.add_argument("--keep-zip", action="store_true", help="Keep downloaded zip files after extraction.")
    parser.add_argument("--no-extract", action="store_true", help="Download zip files but do not extract them.")
    parser.add_argument("--force", action="store_true", help="Download even if a local archive exists.")
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024)
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args()


def human_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown size"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024.0
    return f"{num_bytes}B"


def remote_size(url: str, timeout: int) -> int | None:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = response.headers.get("Content-Length")
            return int(value) if value else None
    except Exception:
        return None


def download_file(asset: Asset, output_dir: Path, args: argparse.Namespace) -> Path:
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / asset.filename
    expected_size = remote_size(asset.url, args.timeout)

    print(f"\nDownloading {asset.description}")
    print(f"  URL: {asset.url}")
    print(f"  To:  {output_path}")
    print(f"  Expected: {human_bytes(expected_size)}")

    if output_path.exists() and not args.force:
        if expected_size is None or output_path.stat().st_size == expected_size:
            print(f"  Exists, skipping: {output_path}")
            return output_path
        print("  Existing file has a different size; downloading again.")

    part_path = output_path.with_suffix(output_path.suffix + ".part")
    if part_path.exists():
        part_path.unlink()

    request = urllib.request.Request(asset.url, headers={"User-Agent": USER_AGENT})
    started = time.monotonic()
    downloaded = 0
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        total = expected_size
        with part_path.open("wb") as out_file:
            while True:
                chunk = response.read(args.chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                print_progress(downloaded, total, started)
    print()

    if expected_size is not None and downloaded != expected_size:
        raise RuntimeError(f"Downloaded size mismatch for {asset.filename}")
    part_path.replace(output_path)
    print(f"  Saved: {output_path}")
    return output_path


def print_progress(downloaded: int, total: int | None, started: float) -> None:
    elapsed = max(time.monotonic() - started, 1e-6)
    rate = downloaded / elapsed
    if total:
        message = f"  {downloaded / total * 100:6.2f}%  {human_bytes(downloaded)} / {human_bytes(total)}  {human_bytes(int(rate))}/s"
    else:
        message = f"  {human_bytes(downloaded)}  {human_bytes(int(rate))}/s"
    print("\r" + message, end="", flush=True)


def safe_extract_zip(zip_path: Path, output_dir: Path, target_dir: Path) -> None:
    if target_dir.exists() and any(target_dir.rglob("*")):
        print(f"  Already extracted: {target_dir}")
        return

    print(f"  Extracting to: {output_dir}")
    output_root = output_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        for member in members:
            target = (output_dir / member.filename).resolve()
            if os.path.commonpath([output_root, target]) != str(output_root):
                raise RuntimeError(f"Unsafe zip member path: {member.filename}")
        for index, member in enumerate(members, start=1):
            archive.extract(member, output_dir)
            if index % 2000 == 0 or index == len(members):
                print(f"\r  Extracted {index}/{len(members)} files", end="", flush=True)
    print()


def image_filename(subset: str, split: str, size: str) -> str:
    size_str = f"{size}p" if size != "fullsize" else "fullsize"
    if subset == "full":
        return f"FungiTastic-{split}-{size_str}.zip"
    return f"FungiTastic-Mini-{split}-{size_str}.zip"


def assets(args: argparse.Namespace) -> list[Asset]:
    output_dir = args.output_dir
    subset_dir = "FungiTastic" if args.subset == "full" else "FungiTastic-Mini"
    out = [
        Asset(
            key="metadata",
            filename="metadata.zip",
            target_dir=output_dir / "metadata" / subset_dir,
            description="FungiTastic metadata CSVs",
        )
    ]
    if args.metadata_only:
        return out

    splits = ["train", "val", "test"]
    if args.include_dna_test:
        splits.append("dna-test")
    for split in splits:
        out.append(
            Asset(
                key=f"mini_{split}_{args.size}",
                filename=image_filename(args.subset, split, args.size),
                target_dir=output_dir / subset_dir / split / (f"{args.size}p" if args.size != "fullsize" else "fullsize"),
                description=f"{subset_dir} {split} images at {args.size}px",
            )
        )
    return out


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("FungiTastic downloader")
    print(f"Output dir: {args.output_dir}")
    print("Default avoids the full ~50GB Kaggle archive and uses Mini 300px images.")
    if args.subset == "full" and args.size == "300":
        print("Full 300px split is still large: train/val/test are about 13.4GB compressed.")

    for asset in assets(args):
        zip_path = download_file(asset, args.output_dir, args)
        if not args.no_extract:
            extract_root = args.output_dir / "metadata" if asset.key == "metadata" else args.output_dir
            safe_extract_zip(zip_path, extract_root, asset.target_dir)
            if not args.keep_zip:
                zip_path.unlink()
                print(f"  Removed archive: {zip_path}")

    print("\nDone.")
    print("You can run:")
    print("  uv run python experiments/fungitastic_toxicity_experiment.py --config_path configs/fungitastic_toxicity_experiment.yaml")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise SystemExit(1)
