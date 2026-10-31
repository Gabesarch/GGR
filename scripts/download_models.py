#!/usr/bin/env python3
"""Download the validation-selected release checkpoints from Hugging Face."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    release = json.loads((ROOT / "configs/release.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", action="append", choices=release["conditions"])
    parser.add_argument("--namespace", default=release["namespace"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--revision", help="Override the pinned Hub revision")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    for key in args.condition or ["ggr-rl"]:
        entry = release["conditions"][key]
        repo_id = args.namespace + "/" + entry["model_id"].split("/", 1)[1]
        revision = args.revision or (entry.get("revision") if args.namespace == release["namespace"] else None)
        path = snapshot_download(repo_id, revision=revision, local_dir=args.output_dir / key)
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
