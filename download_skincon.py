import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm


SKINCON_FITZ_ANNOTATIONS_URL = "https://skincon-dataset.github.io/files/annotations_fitzpatrick17k.csv"
SKINCON_DDI_ANNOTATIONS_URL = "https://skincon-dataset.github.io/files/annotations_ddi.csv"
FITZPATRICK17K_METADATA_URL = "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SkinCon annotation CSVs and Fitzpatrick17k metadata.")
    parser.add_argument("--output_dir", type=str, default="data/skincon")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--download_images", action="store_true", help="Download Fitzpatrick17k source images used by SkinCon.")
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--image_workers", type=int, default=24)
    parser.add_argument("--image_timeout", type=float, default=8.0)
    parser.add_argument("--image_retries", type=int, default=2)
    return parser.parse_args()


def download_file(url: str, path: Path, timeout: float, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        print(f"Already exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {path}")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def skincon_fitzpatrick_rows(output_dir: Path) -> pd.DataFrame:
    annotations = pd.read_csv(output_dir / "annotations_fitzpatrick17k.csv")
    metadata = pd.read_csv(output_dir / "fitzpatrick17k.csv")
    annotations["md5hash"] = annotations["ImageID"].astype(str).str.replace(r"\.[^.]+$", "", regex=True)
    if "Do not consider this image" in annotations.columns:
        annotations = annotations.loc[annotations["Do not consider this image"].fillna(0).astype(int).eq(0)].copy()
    return annotations.merge(metadata, on="md5hash", how="inner", validate="many_to_one")


def download_image(row: dict[str, object], image_dir: Path, timeout: float, retries: int, overwrite: bool) -> dict[str, object]:
    image_id = str(row["ImageID"])
    url = str(row["url"])
    path = image_dir / image_id
    if path.exists() and not overwrite:
        return {"ImageID": image_id, "url": url, "image_path": str(path), "status": "exists", "error": ""}

    last_error = ""
    for _ in range(max(1, retries)):
        try:
            response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            content = response.content
            image = Image.open(BytesIO(content)).convert("RGB")
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path, format="JPEG", quality=95)
            return {"ImageID": image_id, "url": url, "image_path": str(path), "status": "downloaded", "error": ""}
        except Exception as exc:
            last_error = str(exc)
    return {"ImageID": image_id, "url": url, "image_path": str(path), "status": "failed", "error": last_error}


def download_skincon_images(args: argparse.Namespace, output_dir: Path) -> None:
    image_dir = Path(args.image_dir) if args.image_dir else output_dir / "fitzpatrick17k_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    df = skincon_fitzpatrick_rows(output_dir)
    rows = df.to_dict("records")
    records = []
    with ThreadPoolExecutor(max_workers=args.image_workers) as executor:
        futures = [
            executor.submit(download_image, row, image_dir, args.image_timeout, args.image_retries, args.overwrite)
            for row in rows
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="SkinCon image download", unit="image"):
            records.append(future.result())
    manifest = pd.DataFrame(records).sort_values("ImageID")
    manifest.to_csv(output_dir / "fitzpatrick17k_image_download_manifest.csv", index=False)
    print(manifest["status"].value_counts(dropna=False).to_string())
    print(f"Image manifest: {output_dir / 'fitzpatrick17k_image_download_manifest.csv'}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    download_file(SKINCON_FITZ_ANNOTATIONS_URL, output_dir / "annotations_fitzpatrick17k.csv", args.timeout, args.overwrite)
    download_file(SKINCON_DDI_ANNOTATIONS_URL, output_dir / "annotations_ddi.csv", args.timeout, args.overwrite)
    download_file(FITZPATRICK17K_METADATA_URL, output_dir / "fitzpatrick17k.csv", args.timeout, args.overwrite)
    if args.download_images:
        download_skincon_images(args, output_dir)


if __name__ == "__main__":
    main()
