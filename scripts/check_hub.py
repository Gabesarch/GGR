#!/usr/bin/env python3
"""Read-only check of required models and datasets on Hugging Face."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="gsarch")
    args = parser.parse_args()
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError
    release = json.loads((Path(__file__).resolve().parents[1] / "configs/release.json").read_text())
    api = HfApi()
    missing = []
    entries = [("model", c["model_id"]) for c in release["conditions"].values()]
    entries += [("dataset", d["repo_id"]) for d in release["datasets"].values()]
    for kind, original in entries:
        repo_id = args.namespace + "/" + original.split("/", 1)[1]
        try:
            info = api.repo_info(repo_id, repo_type=kind)
            files = {f.rfilename for f in info.siblings}
            if kind == 'dataset':
                if not any(f.endswith(('.parquet', '.jsonl', '.arrow')) for f in files) and 'manifest.json' not in files:
                    missing.append(repo_id)
                    print(f'INCOMPLETE {kind} {repo_id} (repository exists; data files are missing)')
                    continue
            print(f"OK {kind} {repo_id} {'private' if info.private else 'public'} {info.sha}")
        except RepositoryNotFoundError:
            missing.append(repo_id)
            print(f"UNAVAILABLE {kind} {repo_id} (missing or not accessible with this login)")
    raise SystemExit(bool(missing))


if __name__ == "__main__":
    main()
