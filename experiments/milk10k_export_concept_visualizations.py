import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont


MONET_COLS = [
    "MONET_ulceration_crust",
    "MONET_hair",
    "MONET_vasculature_vessels",
    "MONET_erythema",
    "MONET_pigmented",
    "MONET_gel_water_drop_fluid_dermoscopy_liquid",
    "MONET_skin_markings_pen_ink_purple_pen",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export concept explanation grids for MILK10K experiments.")
    parser.add_argument("--experiment_dir", type=str, default="output/milk10k_concept_discovery_improved")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_concepts_per_model", type=int, default=6)
    parser.add_argument("--examples_per_concept", type=int, default=4)
    parser.add_argument("--cell_size", type=int, default=156)
    return parser.parse_args()


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def open_image(path: str, size: int) -> Image.Image:
    try:
        image = Image.open(path).convert("RGB")
    except (FileNotFoundError, OSError):
        image = Image.new("RGB", (size, size), color=(25, 25, 25))
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), color=(245, 245, 245))
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def wrap_text(text: str, max_chars: int) -> list[str]:
    words = text.replace("_", " ").split()
    lines: list[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > max_chars:
            if current:
                lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines[:3]


def concept_selectivity(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = []
    for concept_idx in range(scores.shape[1]):
        class_means = []
        for label in sorted(set(labels.tolist())):
            mask = labels == label
            if mask.any():
                class_means.append(float(scores[mask, concept_idx].mean()))
        values.append(max(class_means) - min(class_means) if class_means else 0.0)
    return np.asarray(values, dtype=np.float32)


def standardize(scores: np.ndarray) -> np.ndarray:
    mean = scores.mean(axis=0, keepdims=True)
    std = scores.std(axis=0, keepdims=True)
    return (scores - mean) / np.clip(std, 1e-6, None)


def draw_text_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int], lines: list[str], font: ImageFont.ImageFont) -> None:
    x, y = xy
    for line in lines:
        draw.text((x, y), line, fill=(20, 20, 20), font=font)
        y += 16


def export_grid(
    df: pd.DataFrame,
    scores: np.ndarray,
    concept_names: list[str],
    title: str,
    output_path: Path,
    max_concepts: int,
    examples_per_concept: int,
    cell_size: int,
    force_all: bool = False,
) -> None:
    labels = df["diagnosis_idx"].to_numpy(dtype=np.int64)
    selection_score = concept_selectivity(scores, labels)
    if force_all:
        selected = list(range(min(len(concept_names), max_concepts)))
    else:
        selected = np.argsort(selection_score)[::-1][: min(max_concepts, len(concept_names))].tolist()

    header_h = 54
    label_w = 230
    row_h = cell_size + 42
    width = label_w + examples_per_concept * cell_size
    height = header_h + len(selected) * row_h
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(18)
    label_font = load_font(12)
    small_font = load_font(11)

    draw.rectangle((0, 0, width, header_h), fill=(238, 242, 246))
    draw.text((12, 15), title, fill=(15, 23, 42), font=title_font)

    for row_pos, concept_idx in enumerate(selected):
        y0 = header_h + row_pos * row_h
        draw.rectangle((0, y0, width, y0 + row_h), fill=(248, 250, 252) if row_pos % 2 == 0 else (255, 255, 255))
        name = concept_names[concept_idx]
        label_lines = wrap_text(f"C{concept_idx:03d}: {name}", 24)
        label_lines.append(f"selectivity={selection_score[concept_idx]:.3f}")
        draw_text_box(draw, (12, y0 + 14), label_lines, label_font)

        ranked = np.argsort(scores[:, concept_idx])[::-1][:examples_per_concept]
        for col_pos, sample_idx in enumerate(ranked):
            x0 = label_w + col_pos * cell_size
            image = open_image(str(df.iloc[sample_idx]["resolved_image_path"]), cell_size)
            canvas.paste(image, (x0, y0))
            diagnosis = str(df.iloc[sample_idx].get("diagnosis", ""))
            score = float(scores[sample_idx, concept_idx])
            draw.rectangle((x0, y0 + cell_size - 30, x0 + cell_size, y0 + cell_size), fill=(255, 255, 255))
            draw.text((x0 + 4, y0 + cell_size - 28), diagnosis[:18], fill=(20, 20, 20), font=small_font)
            draw.text((x0 + 4, y0 + cell_size - 14), f"score={score:.2f}", fill=(20, 20, 20), font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def export_model_map(output_path: Path) -> None:
    rows = [
        ("E1 Linear", "DINOv2 global feature", "No native concepts", "Post-hoc only"),
        ("E2 MLP", "DINOv2 global feature", "No native concepts", "Post-hoc only"),
        ("E3 MONET", "7 predefined terms", "Human-named concept scores", "Direct"),
        ("E4 Euclidean", "Patch centroids", "Top images near centroid", "Prototype"),
        ("E5 Spherical", "Cosine patch centroids", "Top images by cosine activation", "Prototype"),
        ("E6 GMM+PCA", "Patch density modes", "Top images by component likelihood", "Distribution"),
        ("E7 Medoid", "Actual patch exemplars", "Prototype is a real patch", "Exemplar"),
        ("E8 Hybrid", "MONET + E4-E7 banks", "Bank-aware sparse contributions", "Composite"),
    ]
    col_w = [120, 210, 250, 130]
    row_h = 52
    header_h = 56
    width = sum(col_w)
    height = header_h + row_h * len(rows)
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(18)
    font = load_font(12)
    draw.rectangle((0, 0, width, header_h), fill=(226, 232, 240))
    draw.text((14, 16), "MILK10K model explainability map", fill=(15, 23, 42), font=title_font)
    headers = ["Model", "Concept source", "Human explanation", "Status"]
    x = 0
    for idx, header in enumerate(headers):
        draw.rectangle((x, header_h - 20, x + col_w[idx], header_h), fill=(203, 213, 225))
        draw.text((x + 8, header_h - 17), header, fill=(15, 23, 42), font=font)
        x += col_w[idx]

    for row_idx, row in enumerate(rows):
        y = header_h + row_idx * row_h
        fill = (248, 250, 252) if row_idx % 2 == 0 else (255, 255, 255)
        draw.rectangle((0, y, width, y + row_h), fill=fill)
        x = 0
        for col_idx, cell in enumerate(row):
            for line_idx, line in enumerate(wrap_text(cell, max(12, col_w[col_idx] // 8))):
                draw.text((x + 8, y + 9 + line_idx * 15), line, fill=(20, 20, 20), font=font)
            draw.line((x, y, x, y + row_h), fill=(226, 232, 240))
            x += col_w[col_idx]
        draw.line((0, y, width, y), fill=(226, 232, 240))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def load_activation(path: Path) -> np.ndarray:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        tensor = payload["activations"]
    else:
        tensor = payload
    return tensor.float().numpy()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir / "explainability_visuals"
    df = pd.read_csv(experiment_dir / "milk10k_dermoscopy_split.csv")

    export_model_map(output_dir / "model_explainability_map.png")

    monet_scores = df[MONET_COLS].fillna(0.0).to_numpy(dtype=np.float32)
    export_grid(
        df,
        monet_scores,
        MONET_COLS,
        "E3 MONET predefined concepts: top images per concept",
        output_dir / "E3_monet_concepts.png",
        max_concepts=len(MONET_COLS),
        examples_per_concept=args.examples_per_concept,
        cell_size=args.cell_size,
        force_all=True,
    )

    banks = {
        "E4": ("Euclidean K-means concepts", experiment_dir / "activations_E4_euclidean_kmeans_k100.pt"),
        "E5": ("Spherical K-means concepts", experiment_dir / "activations_E5_spherical_kmeans_k100.pt"),
        "E6": ("GMM + PCA concepts", experiment_dir / "activations_E6_gmm_pca_k100.pt"),
        "E7": ("Medoid prototype concepts", experiment_dir / "activations_E7_medoid_prototype_k100.pt"),
    }

    hybrid_parts = [standardize(monet_scores)]
    hybrid_names = [f"E3:{name}" for name in MONET_COLS]
    for model_id, (title, path) in banks.items():
        scores = load_activation(path)
        names = [f"{model_id}_concept_{idx:03d}" for idx in range(scores.shape[1])]
        export_grid(
            df,
            scores,
            names,
            f"{model_id} {title}: top selective concepts",
            output_dir / f"{model_id}_concept_examples.png",
            max_concepts=args.max_concepts_per_model,
            examples_per_concept=args.examples_per_concept,
            cell_size=args.cell_size,
        )
        hybrid_parts.append(standardize(scores))
        hybrid_names.extend(names)

    hybrid_scores = np.concatenate(hybrid_parts, axis=1)
    export_grid(
        df,
        hybrid_scores,
        hybrid_names,
        "E8 hybrid concept bank: top selective source concepts",
        output_dir / "E8_hybrid_concept_examples.png",
        max_concepts=args.max_concepts_per_model,
        examples_per_concept=args.examples_per_concept,
        cell_size=args.cell_size,
    )

    print(f"Saved explainability visuals to {output_dir}")


if __name__ == "__main__":
    main()
