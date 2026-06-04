"""
Download the mushroom datasets and collect external species-level toxicity labels.

This script intentionally does not hard-code edible/poisonous labels. It builds a
weak, provenance-rich label table from external sources:

    1. Dataset1 image/species archive from Kaggle.
    2. Dataset2 UCI/Kaggle Mushroom Classification concepts.
    3. GBIF taxonomy matching for Dataset1 species names.
    4. Wikipedia species page Mycomorphbox `howEdible` values.
    5. Wikipedia poisonous/deadly mushroom lists as unsafe evidence.

The output labels are for experiments only. They are not mushroom consumption
advice. Uncertain, conditional, missing, medicinal-only, inedible, or conflicting
species are excluded from the strict binary experiment.

Usage:
    uv run python download_mushroom_data.py
    uv run python download_mushroom_data.py --skip-downloads
    uv run python download_mushroom_data.py --force-labels
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


DATASET1_URL = "https://www.kaggle.com/api/v1/datasets/download/dariobaumberger/combined-kaggle-mushrooms-dataset"
DATASET2_KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/uciml/mushroom-classification"
DATASET2_UCI_URL = "https://archive.ics.uci.edu/static/public/73/mushroom.zip"
GBIF_MATCH_URL = "https://api.gbif.org/v1/species/match"
WIKIPEDIA_PARSE_URL = "https://en.wikipedia.org/w/api.php"

POISONOUS_LIST_PAGE = "List_of_poisonous_mushroom_species"
DEADLY_LIST_PAGE = "List_of_deadly_mushroom_species"

USER_AGENT = "cv-project-mushroom-downloader/1.0 (https://github.com/openai/codex)"


@dataclass(frozen=True)
class DownloadAsset:
    name: str
    url: str
    path: Path
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download mushroom datasets and collect external toxicity labels."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "mushrooms",
        help="Root directory for mushroom data. Default: data/mushrooms",
    )
    parser.add_argument("--dataset1-url", type=str, default=DATASET1_URL)
    parser.add_argument("--dataset2-url", type=str, default=DATASET2_KAGGLE_URL)
    parser.add_argument(
        "--dataset2-fallback-url",
        type=str,
        default=DATASET2_UCI_URL,
        help="Fallback URL for the UCI mushroom zip if the Kaggle mirror fails.",
    )
    parser.add_argument("--skip-downloads", action="store_true", help="Use existing archives if present.")
    parser.add_argument("--force-downloads", action="store_true", help="Download archives even if they exist.")
    parser.add_argument("--force-labels", action="store_true", help="Re-query external label sources.")
    parser.add_argument("--no-gbif", action="store_true", help="Skip GBIF taxonomy normalization.")
    parser.add_argument("--no-wikipedia-pages", action="store_true", help="Skip per-species Wikipedia page lookup.")
    parser.add_argument("--no-wikipedia-lists", action="store_true", help="Skip Wikipedia poisonous/deadly lists.")
    parser.add_argument("--request-sleep", type=float, default=0.25)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024)
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


def request_url(url: str, timeout: int, data: bytes | None = None, headers: dict[str, str] | None = None) -> bytes:
    request_headers = {"User-Agent": USER_AGENT}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(url, data=data, headers=request_headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def remote_size(url: str, timeout: int) -> int | None:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = response.headers.get("Content-Length")
            return int(value) if value is not None else None
    except Exception:
        return None


def download_file(url: str, output_path: Path, description: str, args: argparse.Namespace) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected_size = remote_size(url, args.timeout)

    print(f"\nDownloading {description}")
    print(f"  URL: {url}")
    print(f"  To:  {output_path}")
    print(f"  Expected: {human_bytes(expected_size)}")

    if output_path.exists() and not args.force_downloads:
        if expected_size is None or output_path.stat().st_size == expected_size:
            print(f"  Exists, skipping: {output_path}")
            return output_path
        print(
            "  Existing file size differs "
            f"({human_bytes(output_path.stat().st_size)} != {human_bytes(expected_size)}), downloading again"
        )

    part_path = output_path.with_suffix(output_path.suffix + ".part")
    if part_path.exists():
        part_path.unlink()

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
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
        raise RuntimeError(f"Downloaded size mismatch for {output_path.name}")
    part_path.replace(output_path)
    print(f"  Saved: {output_path}")
    return output_path


def print_progress(downloaded: int, total: int | None, started: float) -> None:
    elapsed = max(time.monotonic() - started, 1e-6)
    rate = downloaded / elapsed
    if total:
        pct = downloaded / total * 100.0
        msg = f"  {pct:6.2f}%  {human_bytes(downloaded)} / {human_bytes(total)}  {human_bytes(int(rate))}/s"
    else:
        msg = f"  {human_bytes(downloaded)}  {human_bytes(int(rate))}/s"
    print("\r" + msg, end="", flush=True)


def safe_extract_member(zip_path: Path, member_names: list[str], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir.resolve()
    written = []
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        for member_name in member_names:
            if member_name not in names:
                continue
            target = (output_dir / Path(member_name).name).resolve()
            if os.path.commonpath([output_root, target]) != str(output_root):
                raise RuntimeError(f"Unsafe zip member path: {member_name}")
            with archive.open(member_name) as src, target.open("wb") as dst:
                dst.write(src.read())
            written.append(target)
    return written


def ensure_dataset2_csv(raw_zip: Path, output_dir: Path) -> Path:
    uci_dir = output_dir / "uci"
    csv_path = uci_dir / "mushrooms.csv"
    if csv_path.exists():
        return csv_path

    written = safe_extract_member(raw_zip, ["mushrooms.csv", "agaricus-lepiota.data"], uci_dir)
    if csv_path.exists():
        return csv_path
    for path in written:
        if path.name == "agaricus-lepiota.data":
            converted = convert_uci_data_to_csv(path, csv_path)
            return converted
    raise RuntimeError(f"Could not find mushrooms.csv or agaricus-lepiota.data in {raw_zip}")


def convert_uci_data_to_csv(data_path: Path, csv_path: Path) -> Path:
    columns = [
        "class",
        "cap-shape",
        "cap-surface",
        "cap-color",
        "bruises",
        "odor",
        "gill-attachment",
        "gill-spacing",
        "gill-size",
        "gill-color",
        "stalk-shape",
        "stalk-root",
        "stalk-surface-above-ring",
        "stalk-surface-below-ring",
        "stalk-color-above-ring",
        "stalk-color-below-ring",
        "veil-type",
        "veil-color",
        "ring-number",
        "ring-type",
        "spore-print-color",
        "population",
        "habitat",
    ]
    with data_path.open("r", encoding="utf-8", newline="") as src, csv_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.writer(dst)
        writer.writerow(columns)
        for row in csv.reader(src):
            writer.writerow(row)
    return csv_path


def extract_species_from_dataset1(zip_path: Path) -> list[str]:
    species = set()
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if not name.startswith("images/") or name.endswith("/"):
                continue
            parts = name.split("/")
            if len(parts) >= 3:
                species.add(parts[1])
    if not species:
        raise RuntimeError(f"No images/<species>/... entries found in {zip_path}")
    return sorted(species)


def save_species_list(species: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(species) + "\n", encoding="utf-8")


def load_existing_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def query_gbif(species: list[str], cache_path: Path, args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    cached = {} if args.force_labels else (load_existing_json(cache_path) or {})
    out: dict[str, dict[str, Any]] = dict(cached)
    if args.no_gbif:
        return out

    for index, name in enumerate(species, start=1):
        if name in out and out[name].get("source") == "gbif":
            continue
        params = urllib.parse.urlencode({"name": name, "kingdom": "Fungi"})
        url = f"{GBIF_MATCH_URL}?{params}"
        try:
            payload = json.loads(request_url(url, args.timeout).decode("utf-8"))
            out[name] = {"source": "gbif", "payload": payload}
        except Exception as exc:
            out[name] = {"source": "gbif", "error": str(exc)}
        print(f"\rGBIF taxonomy matches: {index}/{len(species)}", end="", flush=True)
        time.sleep(args.request_sleep)
    print()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def gbif_row(name: str, gbif: dict[str, dict[str, Any]]) -> dict[str, Any]:
    payload = gbif.get(name, {}).get("payload", {})
    accepted = payload.get("acceptedScientificName") or payload.get("scientificName") or name
    accepted = strip_author_from_scientific_name(accepted)
    rank = payload.get("rank", "")
    if rank not in {"SPECIES", "SUBSPECIES", "VARIETY", "FORM"} or len(accepted.split()) < 2:
        accepted = name
    return {
        "accepted_name": accepted,
        "gbif_usage_key": payload.get("usageKey", ""),
        "gbif_accepted_usage_key": payload.get("acceptedUsageKey", ""),
        "gbif_taxonomic_status": payload.get("status", ""),
        "gbif_rank": payload.get("rank", ""),
        "gbif_match_type": payload.get("matchType", ""),
        "gbif_confidence": payload.get("confidence", ""),
    }


def strip_author_from_scientific_name(name: str) -> str:
    parts = name.split()
    if len(parts) >= 2:
        return " ".join(parts[:2])
    return name


class WikiTableFirstColumnParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_table = False
        self.table_depth = 0
        self.current_table_index = -1
        self.in_tr = False
        self.in_cell = False
        self.cell_index = -1
        self.first_cell_text: list[str] = []
        self.tables: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        if tag == "table" and "wikitable" in (attr.get("class") or ""):
            self.in_table = True
            self.table_depth += 1
            self.current_table_index += 1
            self.tables.append([])
            return
        if self.in_table and tag == "table":
            self.table_depth += 1
        if self.in_table and tag == "tr":
            self.in_tr = True
            self.cell_index = -1
            self.first_cell_text = []
        if self.in_table and self.in_tr and tag in {"td", "th"}:
            self.cell_index += 1
            self.in_cell = self.cell_index == 0 and tag == "td"

    def handle_endtag(self, tag: str) -> None:
        if self.in_table and tag in {"td", "th"}:
            self.in_cell = False
        if self.in_table and tag == "tr":
            text = clean_wiki_cell_text(" ".join(self.first_cell_text))
            name = extract_binomial(text)
            if name:
                self.tables[self.current_table_index].append(name)
            self.in_tr = False
            self.cell_index = -1
            self.first_cell_text = []
        if self.in_table and tag == "table":
            self.table_depth -= 1
            if self.table_depth <= 0:
                self.in_table = False
                self.table_depth = 0

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.first_cell_text.append(data)


def clean_wiki_cell_text(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"\[[^\]]+\]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def extract_binomial(value: str) -> str:
    match = re.search(r"\b([A-Z][a-z]+(?: [a-z][a-z.-]+){1,2})\b", value)
    return match.group(1).replace(".", "").strip() if match else ""


def fetch_wikipedia_tables(page: str, cache_path: Path, args: argparse.Namespace) -> list[list[str]]:
    if cache_path.exists() and not args.force_labels:
        return load_existing_json(cache_path) or []

    params = urllib.parse.urlencode(
        {
            "action": "parse",
            "page": page,
            "prop": "text",
            "format": "json",
            "formatversion": "2",
        }
    )
    url = f"{WIKIPEDIA_PARSE_URL}?{params}"
    payload = json.loads(request_url(url, args.timeout).decode("utf-8"))
    parser = WikiTableFirstColumnParser()
    parser.feed(payload.get("parse", {}).get("text", ""))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(parser.tables, indent=2, ensure_ascii=False), encoding="utf-8")
    return parser.tables


def collect_wikipedia_evidence(cache_dir: Path, args: argparse.Namespace) -> dict[str, dict[str, str]]:
    if args.no_wikipedia_lists:
        return {}

    evidence: dict[str, dict[str, str]] = {}
    poison_tables = fetch_wikipedia_tables(POISONOUS_LIST_PAGE, cache_dir / "wikipedia_poisonous_tables.json", args)
    deadly_tables = fetch_wikipedia_tables(DEADLY_LIST_PAGE, cache_dir / "wikipedia_deadly_tables.json", args)

    if poison_tables:
        for name in poison_tables[0]:
            evidence[name] = {
                "status": "poisonous",
                "source": f"https://en.wikipedia.org/wiki/{POISONOUS_LIST_PAGE}",
                "source_label": "Wikipedia poisonous mushroom list",
            }
    if deadly_tables:
        for name in deadly_tables[0]:
            evidence[name] = {
                "status": "deadly_poisonous",
                "source": f"https://en.wikipedia.org/wiki/{DEADLY_LIST_PAGE}",
                "source_label": "Wikipedia deadly mushroom list",
            }
    if len(deadly_tables) > 1:
        for name in deadly_tables[1]:
            evidence.setdefault(
                name,
                {
                    "status": "poisonous",
                    "source": f"https://en.wikipedia.org/wiki/{DEADLY_LIST_PAGE}",
                    "source_label": "Wikipedia isolated-death mushroom list",
                },
            )
    return evidence


def collect_wikipedia_page_evidence(
    species: list[tuple[str, list[str]]],
    cache_path: Path,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    cached = {} if args.force_labels else (load_existing_json(cache_path) or {})
    out: dict[str, dict[str, Any]] = dict(cached)
    if args.no_wikipedia_pages:
        return out

    for index, (name, candidates) in enumerate(species, start=1):
        if name in out:
            continue
        try:
            page = fetch_first_wikipedia_species_page(candidates, args)
            wikitext = page.get("wikitext", "")
            out[name] = {
                "page_title": page.get("title", name),
                "page_id": page.get("pageid", ""),
                "revision_id": page.get("revision_id", ""),
                "revision_timestamp": page.get("revision_timestamp", ""),
                "queried_names": candidates,
                "source": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(str(page.get('title', name)).replace(' ', '_'))}",
                "howedible_values": parse_howedible_values(wikitext),
            }
        except Exception as exc:
            out[name] = {"error": str(exc), "howedible_values": []}
        print(f"\rWikipedia species pages: {index}/{len(species)}", end="", flush=True)
        time.sleep(args.request_sleep)
    print()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def fetch_first_wikipedia_species_page(candidates: list[str], args: argparse.Namespace) -> dict[str, Any]:
    errors = []
    for candidate in candidates:
        try:
            return fetch_wikipedia_species_page(candidate, args)
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError("; ".join(errors))


def fetch_wikipedia_species_page(name: str, args: argparse.Namespace) -> dict[str, Any]:
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "titles": name,
            "prop": "revisions",
            "rvprop": "ids|timestamp|content",
            "rvslots": "main",
            "format": "json",
            "formatversion": "2",
            "redirects": "1",
        }
    )
    payload = json.loads(request_url(f"{WIKIPEDIA_PARSE_URL}?{params}", args.timeout).decode("utf-8"))
    if "error" in payload:
        raise RuntimeError(str(payload["error"]))
    pages = payload.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        raise RuntimeError("Wikipedia page not found")
    page = pages[0]
    revision = (page.get("revisions") or [{}])[0]
    return {
        "title": page.get("title", name),
        "pageid": page.get("pageid", ""),
        "revision_id": revision.get("revid", ""),
        "revision_timestamp": revision.get("timestamp", ""),
        "wikitext": revision.get("slots", {}).get("main", {}).get("content", ""),
    }


def parse_howedible_values(wikitext: str) -> list[str]:
    values = []
    for template in re.finditer(r"\{\{\s*mycomorphbox\b(?P<body>.*?)(?:^\}\}|\Z)", wikitext, re.IGNORECASE | re.MULTILINE | re.DOTALL):
        body = template.group("body")
        pattern = re.compile(r"^\s*\|\s*how\s*edible\s*\d*\s*=\s*([^\n\r|<]+)", re.IGNORECASE | re.MULTILINE)
        for match in pattern.finditer(body):
            raw = normalize_wiki_value(match.group(1))
            if raw:
                values.append(raw)
    return sorted(set(values))


def normalize_wiki_value(value: str) -> str:
    value = re.sub(r"<!--.*?-->", " ", value)
    value = re.sub(r"\{\{.*?\}\}", " ", value)
    value = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", value)
    value = re.sub(r"[^A-Za-z _-]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip().lower().replace("_", " ")
    aliases = {
        "excellent": "choice",
        "choice edible": "choice",
        "edible choice": "choice",
        "not edible": "inedible",
        "non edible": "inedible",
        "toxic": "poisonous",
        "poison": "poisonous",
    }
    return aliases.get(value, value)


def decide_label(
    dataset_label: str,
    accepted_name: str,
    page_evidence: dict[str, Any],
    wikipedia_evidence: dict[str, dict[str, str]],
) -> dict[str, str]:
    wiki = wikipedia_evidence.get(dataset_label) or wikipedia_evidence.get(accepted_name)
    howedible = set(str(v).lower() for v in page_evidence.get("howedible_values", []))

    source_bits = []
    if howedible:
        source_bits.append("Wikipedia species page howEdible: " + "; ".join(sorted(howedible)))
    if wiki:
        source_bits.append(wiki["source_label"])

    if wiki and wiki["status"] == "deadly_poisonous":
        return make_decision("deadly_poisonous", "unsafe", "high", source_bits, "external_unsafe_list")
    if wiki and wiki["status"] == "poisonous":
        return make_decision("poisonous", "unsafe", "high", source_bits, "external_unsafe_list")

    unsafe_map = {
        "deadly": "deadly_poisonous",
        "poisonous": "poisonous",
        "psychoactive": "psychoactive_toxic",
        "allergenic": "allergenic_toxic",
    }
    edible_values = {"choice", "edible"}
    conditional_values = {"edible when cooked", "caution"}
    non_strict_values = {"inedible", "medicinal", "unpalatable", "too hard to eat", "unknown"}
    unsafe_values = set(unsafe_map)

    if howedible and howedible.issubset(edible_values):
        confidence = "high" if "choice" in howedible else "medium"
        return make_decision("edible", "edible", confidence, source_bits, "wikipedia_howedible_strict_edible")
    if howedible and howedible.issubset(unsafe_values):
        if "deadly" in howedible:
            return make_decision("deadly_poisonous", "unsafe", "high", source_bits, "wikipedia_howedible_unsafe")
        if "poisonous" in howedible:
            return make_decision("poisonous", "unsafe", "high", source_bits, "wikipedia_howedible_unsafe")
        label = sorted(howedible)[0]
        return make_decision(unsafe_map[label], "unsafe", "medium", source_bits, "wikipedia_howedible_unsafe")
    if (howedible & edible_values) and (howedible & unsafe_values):
        return make_decision("unknown_or_conflicting", "exclude", "medium", source_bits, "mixed_edible_and_unsafe")
    if howedible & conditional_values:
        return make_decision("edible_with_conditions", "exclude", "medium", source_bits, "conditional_or_caution")
    if howedible & non_strict_values:
        return make_decision("inedible_not_known_toxic", "exclude", "medium", source_bits, "non_culinary_or_inedible")
    if howedible:
        return make_decision("unknown_or_conflicting", "exclude", "low", source_bits, "unmapped_or_conflicting_wikipedia_howedible")
    return make_decision("unknown_or_missing", "exclude", "low", source_bits, "no_external_edibility_label")


def make_decision(
    status: str,
    strict_label: str,
    confidence: str,
    source_bits: list[str],
    rule: str,
) -> dict[str, str]:
    evidence = " | ".join(source_bits) if source_bits else "No external edibility/toxicity label found."
    return {
        "edibility_status": status,
        "binary_label_strict": strict_label,
        "confidence": confidence,
        "short_evidence": evidence,
        "label_rule": rule,
    }


def write_label_tables(
    species: list[str],
    gbif: dict[str, dict[str, Any]],
    wikipedia_pages: dict[str, dict[str, Any]],
    wikipedia_evidence: dict[str, dict[str, str]],
    output_dir: Path,
) -> None:
    label_path = output_dir / "species_toxicity_labels.csv"
    provenance_path = output_dir / "species_toxicity_provenance.csv"

    label_fields = [
        "dataset_label",
        "accepted_name",
        "edibility_status",
        "binary_label_strict",
        "confidence",
        "short_evidence",
        "source_hint",
        "label_rule",
    ]
    provenance_fields = label_fields + [
        "wikipedia_page_title",
        "wikipedia_howedible",
        "wikipedia_page_source",
        "wikipedia_page_revision_id",
        "wikipedia_page_revision_timestamp",
        "wikipedia_page_queried_names",
        "wikipedia_page_error",
        "wikipedia_list_source",
        "gbif_usage_key",
        "gbif_accepted_usage_key",
        "gbif_taxonomic_status",
        "gbif_rank",
        "gbif_match_type",
        "gbif_confidence",
        "needs_review",
    ]

    label_rows = []
    provenance_rows = []
    for name in species:
        tax = gbif_row(name, gbif)
        page = wikipedia_pages.get(name, {})
        decision = decide_label(name, tax["accepted_name"], page, wikipedia_evidence)
        wiki = wikipedia_evidence.get(name) or wikipedia_evidence.get(tax["accepted_name"], {})
        source_hint = build_source_hint(decision, page, wiki)
        row = {
            "dataset_label": name,
            "accepted_name": tax["accepted_name"],
            **decision,
            "source_hint": source_hint,
        }
        label_rows.append({field: row.get(field, "") for field in label_fields})

        provenance = {
            **row,
            "wikipedia_page_title": page.get("page_title", ""),
            "wikipedia_howedible": "; ".join(page.get("howedible_values", [])),
            "wikipedia_page_source": page.get("source", ""),
            "wikipedia_page_revision_id": page.get("revision_id", ""),
            "wikipedia_page_revision_timestamp": page.get("revision_timestamp", ""),
            "wikipedia_page_queried_names": "; ".join(page.get("queried_names", [])),
            "wikipedia_page_error": page.get("error", ""),
            "wikipedia_list_source": wiki.get("source", ""),
            **tax,
            "needs_review": "false" if row["binary_label_strict"] in {"edible", "unsafe"} and row["confidence"] == "high" else "true",
        }
        provenance_rows.append({field: provenance.get(field, "") for field in provenance_fields})

    write_csv(label_path, label_fields, label_rows)
    write_csv(provenance_path, provenance_fields, provenance_rows)
    write_summary(label_rows, output_dir / "species_toxicity_collection_summary.json")
    print(f"\nWrote labels:     {label_path}")
    print(f"Wrote provenance: {provenance_path}")


def build_source_hint(decision: dict[str, str], page: dict[str, Any], wiki: dict[str, str]) -> str:
    hints = []
    if page.get("howedible_values"):
        hints.append("Wikipedia howEdible")
    if wiki:
        hints.append(wiki.get("source_label", "Wikipedia list"))
    if not hints:
        hints.append("no external label")
    hints.append(decision["label_rule"])
    return "; ".join(hints)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(label_rows: list[dict[str, str]], path: Path) -> None:
    counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for row in label_rows:
        counts[row["binary_label_strict"]] = counts.get(row["binary_label_strict"], 0) + 1
        status_counts[row["edibility_status"]] = status_counts.get(row["edibility_status"], 0) + 1
    summary = {
        "total_species": len(label_rows),
        "binary_label_strict_counts": counts,
        "edibility_status_counts": status_counts,
        "policy": {
            "edible": "Only Wikipedia species-page howEdible choice/edible with no conflicting unsafe evidence.",
            "unsafe": "Wikipedia species-page howEdible poisonous/deadly/psychoactive/allergenic or Wikipedia poisonous/deadly list.",
            "exclude": "Missing, conditional, inedible, medicinal-only, caution, or conflicting labels.",
        },
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    raw_dir = output_dir / "raw"
    cache_dir = output_dir / "external_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    dataset1_zip = raw_dir / "combined-kaggle-mushrooms-dataset.zip"
    dataset2_zip = raw_dir / "mushroom-classification.zip"

    print("Mushroom dataset downloader and external label collector")
    print(f"Output dir: {output_dir}")

    if not args.skip_downloads:
        download_file(args.dataset1_url, dataset1_zip, "Dataset1 combined Kaggle mushroom images", args)
        try:
            download_file(args.dataset2_url, dataset2_zip, "Dataset2 UCI/Kaggle mushroom concepts", args)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if not args.dataset2_fallback_url:
                raise
            print(f"\nDataset2 primary download failed: {exc}")
            print("Trying UCI fallback URL.")
            download_file(args.dataset2_fallback_url, dataset2_zip, "Dataset2 UCI mushroom concepts fallback", args)
    else:
        if not dataset1_zip.exists():
            raise FileNotFoundError(f"Missing Dataset1 archive: {dataset1_zip}")
        if not dataset2_zip.exists():
            raise FileNotFoundError(f"Missing Dataset2 archive: {dataset2_zip}")

    dataset2_csv = ensure_dataset2_csv(dataset2_zip, output_dir)
    species = extract_species_from_dataset1(dataset1_zip)
    save_species_list(species, output_dir / "species_from_dataset1.txt")
    print(f"\nDataset1 species: {len(species)}")
    print(f"Dataset2 CSV:      {dataset2_csv}")

    gbif = query_gbif(species, cache_dir / "gbif_species_match.json", args)
    page_queries = []
    for name in species:
        accepted_name = gbif_row(name, gbif)["accepted_name"]
        candidates = [name]
        if accepted_name and accepted_name not in candidates:
            candidates.append(accepted_name)
        page_queries.append((name, candidates))
    wikipedia_pages = collect_wikipedia_page_evidence(page_queries, cache_dir / "wikipedia_species_pages.json", args)
    wikipedia_evidence = collect_wikipedia_evidence(cache_dir, args)
    write_label_tables(species, gbif, wikipedia_pages, wikipedia_evidence, output_dir)

    print("\nNext experiment command:")
    print("  uv run python experiments/mushroom_toxicity_experiment.py --config_path configs/mushroom_toxicity_experiment.yaml")
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
