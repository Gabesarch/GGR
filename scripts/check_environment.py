#!/usr/bin/env python3
"""Check installed dependencies, with the release's explicit vLLM/Transformers override."""
import argparse
import importlib.metadata as metadata
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=["eval", "sft", "rl"], required=True)
    args = parser.parse_args()
    installed = {canonicalize_name(d.metadata["Name"]): d for d in metadata.distributions()}
    errors = []
    for name, dist in installed.items():
        for raw in dist.requires or []:
            req = Requirement(raw)
            if req.marker and not req.marker.evaluate({"extra": ""}):
                continue
            dependency = canonicalize_name(req.name)
            found = installed.get(dependency)
            if found is None:
                errors.append(f"{name}: missing {req}")
                continue
            if req.specifier and not req.specifier.contains(found.version, prereleases=True):
                if name == "vllm" and dependency == "transformers" and args.component in {"eval", "rl"}:
                    continue
                errors.append(f"{name}: requires {req}, installed {found.version}")
    if errors:
        raise SystemExit("\n".join(errors))
    import torch
    import transformers
    if args.component in {"eval", "rl"}:
        import vllm
        print("vllm", vllm.__version__)
    if args.component in {"sft", "rl"}:
        import flash_attn
        print("flash_attn", flash_attn.__version__)
    print("torch", torch.__version__, "transformers", transformers.__version__)
    print("Dependency checks passed; CUDA available:", torch.cuda.is_available())


if __name__ == "__main__":
    main()
