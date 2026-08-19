#!/usr/bin/env python3
"""
preprocess_data.py
==================
Step 1B of the NLLB-200 LoRA pipeline.

Reads raw data from data_in/raw/ (output of download_raw_data.py),
concatenates parallel files per language, applies text normalization,
filters blank-line pairs, and writes aligned training/dev pairs to:

    data_in/train/<lang_code>/train.es
    data_in/train/<lang_code>/train.<lang_code>
    data_in/dev/<lang_code>/dev.es
    data_in/dev/<lang_code>/dev.<lang_code>

This script is intentionally SEPARATE from download_raw_data.py so that
you can add extra data (e.g. Awajún) to data_in/raw/ before running this.

Text Normalization (replaces the hacky sed/Asian-char approach):
    1. unicodedata.normalize('NFC', text)  — safe canonical form for all scripts
    2. sacremoses.MosesPunctNormalizer     — curly quotes → straight, em-dash → ' - ',
                                             guillemets → ", whitespace normalization

Usage:
    python scripts/preprocess_data.py                          # all languages
    python scripts/preprocess_data.py --languages specific --lang_list quy,agr,shp

HUMAN REVIEW GATE  ►  After running, spot-check:
    • data_in/train/agr/train.es   (Awajún: Spanish source)
    • data_in/train/agr/train.agr  (Awajún: target)
    • data_in/dev/agr/dev.es
    • Check that line counts match:  wc -l data_in/train/agr/train.*
    • Verify no blank lines remain in any file
"""

import argparse
import unicodedata
from pathlib import Path
from typing import Optional
import sys

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sacremoses import MosesPunctNormalizer


# ---------------------------------------------------------------------------
# Language configuration
# ---------------------------------------------------------------------------
# Each entry defines:
#   nllb_code:      NLLB language token (str)
#   is_native_nllb: whether the token already exists in the base NLLB vocab
#   train_sources:  list of (es_file_rel, lang_file_rel) relative to raw_dir.
#                   Files are concatenated in order.
#   dev_es_rel:     dev Spanish file, relative to raw_dir
#   dev_lang_rel:   dev target file, relative to raw_dir
# File paths are relative to data_in/raw/ (the raw_dir argument).

_BASE = "americasnlp2025/ST1_MachineTranslation/data"
_HELS = "americasnlp2021-st/data"
_REPU = "REPUcs-AmericasNLP2021"

LANGUAGE_CONFIG = {
    # ── Quechua (natively supported by NLLB) ──────────────────────────────
    "quy": {
        "nllb_code": "quy_Latn",
        "is_native_nllb": True,
        "train_sources": [
            (f"{_BASE}/quechua-spanish/parallel_data/es-quy/jw300.es-quy.es",
             f"{_BASE}/quechua-spanish/parallel_data/es-quy/jw300.es-quy.quy"),
            (f"{_BASE}/quechua-spanish/parallel_data/es-quy/minedu.quy-es.es",
             f"{_BASE}/quechua-spanish/parallel_data/es-quy/minedu.quy-es.quy"),
            (f"{_BASE}/quechua-spanish/parallel_data/es-quy/dict_misc.quy-es.es",
             f"{_BASE}/quechua-spanish/parallel_data/es-quy/dict_misc.quy-es.quy"),
            (f"{_BASE}/quechua-spanish/parallel_data/es-quz/jw300.es-quz.es",
             f"{_BASE}/quechua-spanish/parallel_data/es-quz/jw300.es-quz.quz"),
            # Helsinki extra (2021-ST)
            (f"{_HELS}/quechua-spanish/extra/sent-boconst_que.es",
             f"{_HELS}/quechua-spanish/extra/sent-boconst_que.que"),
            (f"{_HELS}/quechua-spanish/extra/sent-peconst.es",
             f"{_HELS}/quechua-spanish/extra/sent-peconst.que"),
            (f"{_HELS}/quechua-spanish/extra/tatoeba_qu.raw.es",
             f"{_HELS}/quechua-spanish/extra/tatoeba_qu.raw.qu"),
            # REPUcs extra
            (f"{_REPU}/Handbook.es",       f"{_REPU}/Handbook.quy"),
            (f"{_REPU}/Lexicon.es",        f"{_REPU}/Lexicon.quy"),
            (f"{_REPU}/WebMisc.es",        f"{_REPU}/WebMisc.quy"),
            (f"{_REPU}/constitucion_simplified.es",  f"{_REPU}/constitucion_simplified.quz"),
            (f"{_REPU}/reglamento_simplified.es",    f"{_REPU}/reglamento_simplified.quz"),
        ],
        "dev_es_rel":   f"{_BASE}/quechua-spanish/dev.es",
        "dev_lang_rel": f"{_BASE}/quechua-spanish/dev.quy",
    },

    # ── Aymara (natively supported) ───────────────────────────────────────
    "ayr": {
        "nllb_code": "ayr_Latn",
        "is_native_nllb": True,
        "train_sources": [
            (f"{_BASE}/aymara-spanish/parallel_data/es-aym/opus_globalvoices.es-aym.es",
             f"{_BASE}/aymara-spanish/parallel_data/es-aym/opus_globalvoices.es-aym.aym"),
            # Helsinki extra (2021-ST)
            (f"{_HELS}/aymara-spanish/extra/jw_aym.es",
             f"{_HELS}/aymara-spanish/extra/jw_aym.aym"),
            (f"{_HELS}/aymara-spanish/extra/sent-boconst_aym.es",
             f"{_HELS}/aymara-spanish/extra/sent-boconst_aym.aym"),
        ],
        "dev_es_rel":   f"{_BASE}/aymara-spanish/dev.es",
        "dev_lang_rel": f"{_BASE}/aymara-spanish/dev.aym",
    },

    # ── Shipibo-Konibo (new token: shp_Latn) ──────────────────────────────
    "shp": {
        "nllb_code": "shp_Latn",
        "is_native_nllb": False,
        "train_sources": [
            # AmericasNLP 2025 base train (if exists)
            (f"{_BASE}/shipibo_konibo-spanish/train.es",
             f"{_BASE}/shipibo_konibo-spanish/train.shp"),
            # Helsinki extra (2021-ST) — Educational domain
            (f"{_HELS}/shipibo_konibo-spanish/extra/Educational_0.4_2.4_35/train-es-shi.es",
             f"{_HELS}/shipibo_konibo-spanish/extra/Educational_0.4_2.4_35/train-es-shi.shi"),
            (f"{_HELS}/shipibo_konibo-spanish/extra/Educational_0.4_2.4_35/test-es-shi.es",
             f"{_HELS}/shipibo_konibo-spanish/extra/Educational_0.4_2.4_35/test-es-shi.shi"),
            (f"{_HELS}/shipibo_konibo-spanish/extra/Educational_0.4_2.4_35/tune-es-shi.es",
             f"{_HELS}/shipibo_konibo-spanish/extra/Educational_0.4_2.4_35/tune-es-shi.shi"),
            # Helsinki extra — Religious domain
            (f"{_HELS}/shipibo_konibo-spanish/extra/Religious_0.2_2.4_35/train-es-shi.es",
             f"{_HELS}/shipibo_konibo-spanish/extra/Religious_0.2_2.4_35/train-es-shi.shi"),
            (f"{_HELS}/shipibo_konibo-spanish/extra/Religious_0.2_2.4_35/test-es-shi.es",
             f"{_HELS}/shipibo_konibo-spanish/extra/Religious_0.2_2.4_35/test-es-shi.shi"),
            (f"{_HELS}/shipibo_konibo-spanish/extra/Religious_0.2_2.4_35/tune-es-shi.es",
             f"{_HELS}/shipibo_konibo-spanish/extra/Religious_0.2_2.4_35/tune-es-shi.shi"),
            # Other extra files
            (f"{_HELS}/shipibo_konibo-spanish/extra/sent-leyartesano.es",
             f"{_HELS}/shipibo_konibo-spanish/extra/sent-leyartesano.shi"),
            (f"{_HELS}/shipibo_konibo-spanish/extra/traduccionTsanas1.es",
             f"{_HELS}/shipibo_konibo-spanish/extra/traduccionTsanas1.shi"),
        ],
        "dev_es_rel":   f"{_BASE}/shipibo_konibo-spanish/dev.es",
        "dev_lang_rel": f"{_BASE}/shipibo_konibo-spanish/dev.shp",
    },

    # ── Awajún (new token: agr_Latn) ──────────────────────────────────────
    "agr": {
        "nllb_code": "agr_Latn",
        "is_native_nllb": False,
        "train_sources": [
            (f"{_BASE}/awajun-spanish/train.es",
             f"{_BASE}/awajun-spanish/train.agr"),
        ],
        "dev_es_rel":   f"{_BASE}/awajun-spanish/dev.es",
        "dev_lang_rel": f"{_BASE}/awajun-spanish/dev.agr",
    },

    # ── Ashaninka (new token: cni_Latn) ───────────────────────────────────
    "cni": {
        "nllb_code": "cni_Latn",
        "is_native_nllb": False,
        "train_sources": [
            (f"{_BASE}/ashaninka-spanish/train.es",
             f"{_BASE}/ashaninka-spanish/train.cni"),
        ],
        "dev_es_rel":   f"{_BASE}/ashaninka-spanish/dev.es",
        "dev_lang_rel": f"{_BASE}/ashaninka-spanish/dev.cni",
    },

    # ── Guaraní — grn (natively supported) ────────────────────────────────
    "grn": {
        "nllb_code": "grn_Latn",
        "is_native_nllb": True,
        "train_sources": [
            # AmericasNLP 2025 base train
            (f"{_BASE}/guarani-spanish/train.es",
             f"{_BASE}/guarani-spanish/train.gn"),
            # Helsinki extra (2021-ST)
            (f"{_HELS}/guarani-spanish/extra/sent-pyconst.es",
             f"{_HELS}/guarani-spanish/extra/sent-pyconst.gn"),
        ],
        "dev_es_rel":   f"{_BASE}/guarani-spanish/dev.es",
        "dev_lang_rel": f"{_BASE}/guarani-spanish/dev.gn",
    },

    # ── Bribri (new token: bzd_Latn) ─────────────────────────────────────
    "bzd": {
        "nllb_code": "bzd_Latn",
        "is_native_nllb": False,
        "train_sources": [
            (f"{_BASE}/bribri-spanish/train.es",
             f"{_BASE}/bribri-spanish/train.bzd"),
        ],
        "dev_es_rel":   f"{_BASE}/bribri-spanish/dev.es",
        "dev_lang_rel": f"{_BASE}/bribri-spanish/dev.bzd",
    },

    # ── Wayuu (new token: guc_Latn) ───────────────────────────────────────
    "guc": {
        "nllb_code": "guc_Latn",
        "is_native_nllb": False,
        "train_sources": [
            (f"{_BASE}/wayuu-spanish/train.es",
             f"{_BASE}/wayuu-spanish/train.guc"),
        ],
        "dev_es_rel":   f"{_BASE}/wayuu-spanish/dev.es",
        "dev_lang_rel": f"{_BASE}/wayuu-spanish/dev.guc",
    },

    # ── Nahuatl (new token: nah_Latn) ─────────────────────────────────────
    "nah": {
        "nllb_code": "nah_Latn",
        "is_native_nllb": False,
        "train_sources": [
            # AmericasNLP 2025
            (f"{_BASE}/nahuatl-spanish/train.es",
             f"{_BASE}/nahuatl-spanish/train.nah"),
            # Helsinki extra (2021-ST)
            (f"{_HELS}/nahuatl-spanish/extra/mxconst.es",
             f"{_HELS}/nahuatl-spanish/extra/mxconst.nah"),
            (f"{_HELS}/nahuatl-spanish/extra/sent-mxconst.es",
             f"{_HELS}/nahuatl-spanish/extra/sent-mxconst.nah"),
        ],
        "dev_es_rel":   f"{_BASE}/nahuatl-spanish/dev.es",
        "dev_lang_rel": f"{_BASE}/nahuatl-spanish/dev.nah",
    },

    # ── Hñähñu / Otomi (new token: oto_Latn) ─────────────────────────────
    "oto": {
        "nllb_code": "oto_Latn",
        "is_native_nllb": False,
        "train_sources": [
            # Helsinki extra (2021-ST)
            (f"{_HELS}/hñähñu-spanish/extra/mxconst.es",
             f"{_HELS}/hñähñu-spanish/extra/mxconst.oto"),
            (f"{_HELS}/hñähñu-spanish/extra/sent-mxconst.es",
             f"{_HELS}/hñähñu-spanish/extra/sent-mxconst.oto"),
        ],
        "dev_es_rel":   f"{_BASE}/otomi-spanish/dev.es",
        "dev_lang_rel": f"{_BASE}/otomi-spanish/dev.oto",
    },

    # ── Rarámuri / Tarahumara (new token: tar_Latn) ───────────────────────
    "tar": {
        "nllb_code": "tar_Latn",
        "is_native_nllb": False,
        "train_sources": [
            # Helsinki extra (2021-ST)
            (f"{_HELS}/raramuri-spanish/extra/mxconst.es",
             f"{_HELS}/raramuri-spanish/extra/mxconst.tar"),
            (f"{_HELS}/raramuri-spanish/extra/sent-mxconst.es",
             f"{_HELS}/raramuri-spanish/extra/sent-mxconst.tar"),
        ],
        "dev_es_rel":   f"{_BASE}/raramuri-spanish/dev.es",
        "dev_lang_rel": f"{_BASE}/raramuri-spanish/dev.tar",
    },

    # ── Wixarika / Huichol (new token: hch_Latn) ─────────────────────────
    "hch": {
        "nllb_code": "hch_Latn",
        "is_native_nllb": False,
        "train_sources": [
            # Helsinki extra (2021-ST)
            (f"{_HELS}/wixarika-spanish/extra/corpora.es",
             f"{_HELS}/wixarika-spanish/extra/corpora.wix"),
            (f"{_HELS}/wixarika-spanish/extra/sent-mxconst.es",
             f"{_HELS}/wixarika-spanish/extra/sent-mxconst.hch"),
            (f"{_HELS}/wixarika-spanish/extra/paral_own.es",
             f"{_HELS}/wixarika-spanish/extra/paral_own.wix"),
        ],
        "dev_es_rel":   f"{_BASE}/wixarika-spanish/dev.es",
        "dev_lang_rel": f"{_BASE}/wixarika-spanish/dev.hch",
    },

    # ── Chatino (new token: ctp_Latn) ─────────────────────────────────────
    "ctp": {
        "nllb_code": "ctp_Latn",
        "is_native_nllb": False,
        "train_sources": [
            (f"{_BASE}/chatino-spanish/train.es",
             f"{_BASE}/chatino-spanish/train.ctp"),
        ],
        "dev_es_rel":   f"{_BASE}/chatino-spanish/dev.es",
        "dev_lang_rel": f"{_BASE}/chatino-spanish/dev.ctp",
    },
}

ALL_LANGS = list(LANGUAGE_CONFIG.keys())


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

# Cache normalizer instances per language to avoid re-creation per line
_NORMALIZERS: dict[str, MosesPunctNormalizer] = {}


def _get_normalizer(lang: str = "es") -> MosesPunctNormalizer:
    if lang not in _NORMALIZERS:
        _NORMALIZERS[lang] = MosesPunctNormalizer(lang=lang)
    return _NORMALIZERS[lang]


def normalize_text(text: str, lang: str = "es") -> str:
    """
    SOTA text normalization for MT preprocessing.

    Step 1 — NFC Unicode normalization
        Ensures all combining characters are in canonical composed form.
        Safe for all scripts including indigenous languages with diacritics.
        NFC is used (not NFKC) to avoid collapsing legitimate Unicode
        representations used in some indigenous orthographies.

    Step 2 — Moses punctuation normalization (sacremoses)
        Handles typographic/Unicode punctuation → ASCII equivalents:
        - Curly double quotes (" ") → " (straight ASCII quote)
        - Curly single quotes (' ') → ' (straight ASCII apostrophe)
        - Em-dash (—) → ' - ' (with spaces — avoids fusing adjacent tokens)
        - En-dash (–) → - (bare hyphen, short enough to not need spaces)
        - Guillemets (« ») → " (ASCII quote — not << >> which NLLB can't handle)
        - Baseline-9 quote (‚) → , (comma)
        - Unicode whitespace variants → ASCII space
        - Multiple spaces → single space
    """
    # Step 1: NFC
    text = unicodedata.normalize("NFC", text)
    # Step 2: Moses
    normalizer = _get_normalizer(lang)
    text = normalizer.normalize(text)
    return text.strip()


# ---------------------------------------------------------------------------
# Core processing helpers
# ---------------------------------------------------------------------------

def read_lines(path: Path) -> list[str]:
    """Read a text file, returning lines with newlines stripped."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def filter_and_normalize(
    es_lines: list[str],
    lang_lines: list[str],
    lang_code: str,
    verbose: bool = True,
) -> tuple[list[str], list[str]]:
    """
    1. Remove pairs where either side is blank.
    2. Apply text normalization to both sides.
    Returns (es_filtered, lang_filtered).
    """
    total = len(es_lines)
    assert total == len(lang_lines), (
        f"Line count mismatch: {total} es vs {len(lang_lines)} {lang_code}"
    )

    es_out, lang_out = [], []
    skipped = 0
    for es, lang in zip(es_lines, lang_lines):
        if not es.strip() or not lang.strip():
            skipped += 1
            continue
        es_out.append(normalize_text(es, lang="es"))
        lang_out.append(normalize_text(lang, lang="es"))  # Moses has no model for indigenous langs
    if verbose:
        print(f"      Filtered {skipped}/{total} blank pairs → {len(es_out)} pairs kept")
    return es_out, lang_out


def write_pair(es_lines: list[str], lang_lines: list[str], out_dir: Path, split: str, lang_code: str) -> None:
    """Write aligned es and lang files to out_dir/<split>/<lang_code>/."""
    out_subdir = out_dir / split / lang_code
    out_subdir.mkdir(parents=True, exist_ok=True)
    es_path = out_subdir / f"{split}.es"
    lang_path = out_subdir / f"{split}.{lang_code}"
    with open(es_path, "w", encoding="utf-8") as f:
        f.write("\n".join(es_lines) + "\n")
    with open(lang_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lang_lines) + "\n")
    print(f"      Wrote {len(es_lines):>6,} pairs → {es_path.parent}/{split}.{{es,{lang_code}}}")


# ---------------------------------------------------------------------------
# Per-language processing
# ---------------------------------------------------------------------------

def process_language(lang_code: str, raw_dir: Path, out_dir: Path) -> dict:
    """Process one language: build train + dev aligned files."""
    cfg = LANGUAGE_CONFIG[lang_code]
    nllb_code = cfg["nllb_code"]
    tag = "[native]" if cfg["is_native_nllb"] else "[new]   "
    print(f"\n  ── {lang_code.upper()} ({nllb_code}) {tag} ──")

    stats = {"lang": lang_code, "nllb_code": nllb_code, "train_pairs": 0, "dev_pairs": 0, "missing_sources": []}

    # ----- TRAIN --------------------------------------------------------
    all_es, all_lang = [], []
    for es_rel, lang_rel in cfg["train_sources"]:
        es_path = raw_dir / es_rel
        lang_path = raw_dir / lang_rel
        if not es_path.exists():
            print(f"      [SKIP] Missing: {es_path.name}")
            stats["missing_sources"].append(str(es_path))
            continue
        if not lang_path.exists():
            print(f"      [SKIP] Missing: {lang_path.name}")
            stats["missing_sources"].append(str(lang_path))
            continue
        es_lines = read_lines(es_path)
        lang_lines = read_lines(lang_path)
        if len(es_lines) != len(lang_lines):
            min_len = min(len(es_lines), len(lang_lines))
            print(f"      [WARN] {es_path.name}: length mismatch "
                  f"({len(es_lines)} vs {len(lang_lines)}), truncating to {min_len}")
            es_lines, lang_lines = es_lines[:min_len], lang_lines[:min_len]
        all_es.extend(es_lines)
        all_lang.extend(lang_lines)
        print(f"      + {es_path.name}  ({len(es_lines):,} lines)")

    if all_es:
        es_filtered, lang_filtered = filter_and_normalize(all_es, all_lang, lang_code)
        write_pair(es_filtered, lang_filtered, out_dir, "train", lang_code)
        stats["train_pairs"] = len(es_filtered)
    else:
        print(f"      [WARN] No training data found for {lang_code}!")

    # ----- DEV ----------------------------------------------------------
    dev_es_path = raw_dir / cfg["dev_es_rel"]
    dev_lang_path = raw_dir / cfg["dev_lang_rel"]
    if dev_es_path.exists() and dev_lang_path.exists():
        dev_es = read_lines(dev_es_path)
        dev_lang = read_lines(dev_lang_path)
        if len(dev_es) != len(dev_lang):
            min_len = min(len(dev_es), len(dev_lang))
            dev_es, dev_lang = dev_es[:min_len], dev_lang[:min_len]
        dev_es_f, dev_lang_f = filter_and_normalize(dev_es, dev_lang, lang_code, verbose=True)
        write_pair(dev_es_f, dev_lang_f, out_dir, "dev", lang_code)
        stats["dev_pairs"] = len(dev_es_f)
    else:
        print(f"      [WARN] Dev files not found for {lang_code}")

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess AmericasNLP raw data into per-language aligned files"
    )
    parser.add_argument(
        "--languages",
        choices=["all", "specific"],
        default="all",
        help="Process all languages or a specific subset",
    )
    parser.add_argument(
        "--lang_list",
        type=str,
        default="",
        help="Comma-separated language codes when --languages specific, e.g. 'quy,agr,shp'",
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=None,
        help="Path to raw data directory. Default: <project_root>/data_in/raw",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Root output directory. Default: <project_root>/data_in",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    raw_dir = Path(args.raw_dir) if args.raw_dir else project_root / "data_in" / "raw"
    out_dir = Path(args.output_dir) if args.output_dir else project_root / "data_in"

    # Determine which languages to process
    if args.languages == "all":
        langs = ALL_LANGS
    else:
        langs = [l.strip() for l in args.lang_list.split(",") if l.strip()]
        unknown = [l for l in langs if l not in LANGUAGE_CONFIG]
        if unknown:
            print(f"[ERROR] Unknown language codes: {unknown}")
            print(f"  Valid codes: {ALL_LANGS}")
            raise SystemExit(1)

    print(f"\n{'='*60}")
    print(f"  NLLB-200 LoRA Pipeline — Step 1B: Preprocess Data")
    print(f"  Raw data dir : {raw_dir}")
    print(f"  Output dir   : {out_dir}")
    print(f"  Languages    : {langs}")
    print(f"{'='*60}")

    # Process each language
    all_stats = []
    for lang_code in langs:
        stats = process_language(lang_code, raw_dir, out_dir)
        all_stats.append(stats)

    # Summary table
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Lang':<6} {'NLLB token':<18} {'Train pairs':>12} {'Dev pairs':>10}")
    print(f"  {'-'*6} {'-'*18} {'-'*12} {'-'*10}")
    for s in all_stats:
        native = "" if LANGUAGE_CONFIG[s["lang"]]["is_native_nllb"] else "*"
        print(f"  {s['lang']:<6} {s['nllb_code']:<18} {s['train_pairs']:>12,} {s['dev_pairs']:>10,}  {native}")
    print(f"\n  * = new token (not natively supported by NLLB-200)")
    print(f"\n  ► HUMAN REVIEW GATE: spot-check line counts above and verify")
    print(f"    no language has 0 training pairs unexpectedly.")
    print()


if __name__ == "__main__":
    main()
