#!/usr/bin/env python3
"""
download_raw_data.py
====================
Step 1A of the NLLB-200 LoRA pipeline.

Downloads all raw data repositories and archives to data_in/raw/.
Does NOT preprocess or modify any files — run preprocess_data.py next.

Safe to re-run: already-cloned repos and downloaded archives are skipped.

After running this script you can manually add extra data (e.g. more Awajún
texts) to data_in/raw/americasnlp2025/ST1_MachineTranslation/data/awajun-spanish/
before running preprocess_data.py.

Usage:
    python scripts/download_raw_data.py
    python scripts/download_raw_data.py --data_dir /custom/path/to/raw

HUMAN REVIEW GATE  ►  After this script completes, verify that:
    • data_in/raw/americasnlp2025/        exists and is non-empty
    • data_in/raw/americasnlp2021-st/     exists and is non-empty
    • data_in/raw/REPUcs-AmericasNLP2021/ exists and is non-empty
    • data_in/raw/nllb_seed_data/         directory exists with NLLB-Seed files
"""

import argparse
import os
import subprocess
import sys

# Force UTF-8 output — required on Windows (cp1252 can't encode box-drawing chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import zipfile
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Repos to clone  {local_subdir: git_url}
GIT_REPOS = {
    "americasnlp2025": "https://github.com/AmericasNLP/americasnlp2025.git",
    "americasnlp2021-st": "https://github.com/Helsinki-NLP/americasnlp2021-st.git",
    "REPUcs-AmericasNLP2021": "https://github.com/Ceviche98/REPUcs-AmericasNLP2021.git",
}

# Zip archives to download  {local_subdir: (url, zip_filename)}
# NOTE: NLLB-MD (chat/news/health) is intentionally excluded.
#       Add it back here once you verify AmericasNLP data overlap.
ZIP_DOWNLOADS = {
    "nllb_seed_data": [
        ("https://tinyurl.com/NLLBSeed", "NLLBSeed.zip"),
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], cwd: Path | None = None) -> None:
    """Run a subprocess command, printing it first. Raises on failure."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    """Download url → dest with a progress indicator."""
    print(f"  Downloading {url}  →  {dest.name}")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  Progress: {pct:.1f}%", end="", flush=True)
    print()  # newline after progress display


def unzip(archive: Path, target_dir: Path) -> None:
    """Extract a zip archive into target_dir."""
    print(f"  Extracting {archive.name}  →  {target_dir}")
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(target_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download AmericasNLP 2025 raw data to data_in/raw/"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory to download into. Default: <project_root>/data_in/raw",
    )
    args = parser.parse_args()

    # Determine project root (one level up from scripts/)
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = Path(args.data_dir) if args.data_dir else project_root / "data_in" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  NLLB-200 LoRA Pipeline — Step 1A: Download Raw Data")
    print(f"  Output directory: {raw_dir}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1.  Clone git repositories
    # ------------------------------------------------------------------
    print("-- Git repositories ------------------------------------------")
    for subdir, url in GIT_REPOS.items():
        dest = raw_dir / subdir
        if dest.exists() and any(dest.iterdir()):
            print(f"  [SKIP] {subdir} already exists at {dest}")
        else:
            print(f"  Cloning {url}")
            run(["git", "clone", "--depth", "1", url, str(dest)])
            print(f"  [OK]   {subdir}")
    print()

    # ------------------------------------------------------------------
    # 2.  Download & unzip archives
    # ------------------------------------------------------------------
    print("-- Zip archives ----------------------------------------------")
    for subdir, files in ZIP_DOWNLOADS.items():
        dest_dir = raw_dir / subdir
        dest_dir.mkdir(parents=True, exist_ok=True)

        for url, zip_name in files:
            zip_path = dest_dir / zip_name
            # Check if already extracted (look for any non-zip file in dest_dir)
            extracted = [f for f in dest_dir.rglob("*") if f.is_file() and f.suffix != ".zip"]
            if extracted:
                print(f"  [SKIP] {zip_name} already extracted ({len(extracted)} files found)")
                continue

            # Download
            if not zip_path.exists():
                download_file(url, zip_path)
            else:
                print(f"  [SKIP] {zip_name} archive already present, re-extracting...")

            # Unzip
            unzip(zip_path, dest_dir)
            print(f"  [OK]   {zip_name} → {dest_dir}")
    print()

    # ------------------------------------------------------------------
    # 3.  Summary
    # ------------------------------------------------------------------
    print("-- Download complete -----------------------------------------")
    for item in sorted(raw_dir.iterdir()):
        if item.is_dir():
            n_files = sum(1 for _ in item.rglob("*") if _.is_file())
            print(f"  {item.name}/ ({n_files} files)")
    print()
    print("Next step:")
    print("  python scripts/preprocess_data.py --languages all")
    print()
    print("  ► HUMAN REVIEW GATE: verify the directories above are non-empty")
    print("    and add any extra Awajún data to:")
    print(f"    {raw_dir / 'americasnlp2025' / 'ST1_MachineTranslation' / 'data' / 'awajun-spanish'}")
    print("    before running preprocess_data.py.")
    print()


if __name__ == "__main__":
    main()
