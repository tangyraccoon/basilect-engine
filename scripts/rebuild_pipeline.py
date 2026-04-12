"""Rebuild all pipeline outputs (embeddings, matrices, discoveries) from artist JSON data.

Run this after cloning the repo or when .npy / discoveries files are missing.
Does NOT re-scrape or re-extract — it only regenerates the numeric pipeline outputs
from the per-artist quotes.json and critic_quotes.json that are already in the repo.

Usage:
    python scripts/rebuild_pipeline.py [--signals interview critic both] [--min-quotes N]
"""

import subprocess
import sys
from pathlib import Path


def run(cmd: list, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent)
    if result.returncode != 0:
        print(f"  ERROR: {label} failed (exit {result.returncode})")
        sys.exit(result.returncode)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Rebuild embeddings, matrices, and discoveries")
    parser.add_argument("--signals", choices=["interview", "critic", "both"], default="both",
                        help="Which signal(s) to rebuild (default: both)")
    args = parser.parse_args()

    scripts = Path(__file__).resolve().parent

    if args.signals in ("interview", "both"):
        run([sys.executable, str(scripts / "embed.py")], "Embed interview quotes")
        run([sys.executable, str(scripts / "compute.py")], "Compute interview similarity matrix")
        run([sys.executable, str(scripts / "discover.py")], "Generate interview discoveries")

    if args.signals in ("critic", "both"):
        run([sys.executable, str(scripts / "embed_critics.py")], "Embed critic quotes")
        run([sys.executable, str(scripts / "compute_critics.py")], "Compute critic similarity matrix")
        run([sys.executable, str(scripts / "discover_critics.py")], "Generate critic discoveries")

    print("\nDone. Pipeline outputs rebuilt.")
    print("  data/embeddings.npy")
    print("  data/critic_embeddings.npy")
    print("  data/similarity.npy")
    print("  data/critic_similarity.npy")
    print("  data/discoveries.json")
    print("  data/critic_discoveries.json")


if __name__ == "__main__":
    main()
