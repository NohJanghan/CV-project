import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export top activating FungiTastic concept examples.")
    parser.add_argument("--experiment_dir", type=str, default="output/fungitastic_toxicity")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_id", type=str, default="F3", choices=["F3", "F4"])
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--max_concepts", type=int, default=8)
    parser.add_argument("--examples_per_concept", type=int, default=4)
    parser.add_argument("--cell_size", type=int, default=156)
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def concept_selectivity(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = []
    for i in range(scores.shape[1]):
        edible = scores[labels == 0, i]
        unsafe = scores[labels == 1, i]
        if len(edible) == 0 or len(unsafe) == 0:
            out.append(0.0)
        else:
            out.append(abs(float(unsafe.mean() - edible.mean())))
    return np.asarray(out)


def open_image(path: str, size: int) -> Image.Image:
    try:
        image = Image.open(path).convert("RGB")
    except Exception:
        image = Image.new("RGB", (size, size), color=(25, 25, 25))
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), color=(245, 245, 245))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir / "explainability_visuals"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(experiment_dir / "manifest.csv")
    if args.model_id == "F4":
        payload_path = experiment_dir / f"fungitastic_medoids_k{args.k}.pt"
        title = "F4 FungiTastic medoid prototype concepts: top activating examples"
    else:
        payload_path = experiment_dir / f"fungitastic_concepts_k{args.k}.pt"
        title = "F3 FungiTastic learned visual concepts: top activating examples"
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    scores = payload["activations"].float().numpy()
    labels = df["target"].to_numpy(dtype=np.int64)
    selectivity = concept_selectivity(scores, labels)
    selected = np.argsort(selectivity)[::-1][: args.max_concepts].tolist()

    cell = args.cell_size
    label_w = 250
    header_h = 58
    row_h = cell + 48
    width = label_w + args.examples_per_concept * cell
    height = header_h + len(selected) * row_h
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(18)
    label_font = load_font(12)
    small_font = load_font(11)
    draw.rectangle((0, 0, width, header_h), fill=(226, 232, 240))
    draw.text((14, 18), title, fill=(15, 23, 42), font=title_font)

    for row_pos, concept_idx in enumerate(selected):
        y0 = header_h + row_pos * row_h
        draw.rectangle((0, y0, width, y0 + row_h), fill=(248, 250, 252) if row_pos % 2 == 0 else (255, 255, 255))
        unsafe_mean = scores[labels == 1, concept_idx].mean()
        edible_mean = scores[labels == 0, concept_idx].mean()
        lines = [
            f"C{concept_idx:03d}",
            f"selectivity={selectivity[concept_idx]:.3f}",
            f"unsafe_mean={unsafe_mean:.3f}",
            f"edible_mean={edible_mean:.3f}",
        ]
        for i, line in enumerate(lines):
            draw.text((12, y0 + 12 + i * 16), line, fill=(20, 20, 20), font=label_font)
        ranked = np.argsort(scores[:, concept_idx])[::-1][: args.examples_per_concept]
        for col_pos, sample_idx in enumerate(ranked):
            x0 = label_w + col_pos * cell
            image = open_image(df.iloc[sample_idx]["image_path"], cell)
            canvas.paste(image, (x0, y0))
            species = str(df.iloc[sample_idx]["species"])
            label = "unsafe" if int(df.iloc[sample_idx]["target"]) == 1 else "edible"
            score = scores[sample_idx, concept_idx]
            draw.rectangle((x0, y0 + cell - 38, x0 + cell, y0 + cell), fill=(255, 255, 255))
            draw.text((x0 + 4, y0 + cell - 36), species[:24], fill=(20, 20, 20), font=small_font)
            draw.text((x0 + 4, y0 + cell - 20), f"{label} score={score:.2f}", fill=(20, 20, 20), font=small_font)

    out_path = output_dir / f"{args.model_id}_fungitastic_concept_examples.png"
    canvas.save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
