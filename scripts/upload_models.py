#!/usr/bin/env python3
"""Package selected Qwen3-VL release checkpoints; upload only with --upload."""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
    "tokenizer.model", "special_tokens_map.json", "added_tokens.json", "vocab.json",
    "merges.txt", "processor_config.json", "preprocessor_config.json",
    "video_preprocessor_config.json",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def original_template(source):
    if (source / "chat_template.jinja").is_file():
        return (source / "chat_template.jinja").read_text()
    if (source / "chat_template.json").is_file():
        return read_json(source / "chat_template.json")["chat_template"]
    value = read_json(source / "tokenizer_config.json").get("chat_template")
    if not isinstance(value, str) or not value:
        raise ValueError("A single native chat template is required")
    return value


def default_template(native, prompt):
    return (
        "{# GGR condition default system prompt. #}\n"
        "{%- if messages and messages[0]['role'] != 'system' -%}\n"
        "{%- set ggr_system_prompt = " + json.dumps(prompt) + " -%}\n"
        "{%- set messages = [{'role': 'system', 'content': ggr_system_prompt}] + messages -%}\n"
        "{%- endif -%}\n" + native
    )


def preflight(source, entry):
    for name in ["config.json", "tokenizer_config.json", "tokenizer.json"]:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    index = source / "model.safetensors.index.json"
    if index.exists():
        names = sorted(set(read_json(index)["weight_map"].values()))
        if any(Path(n).name != n for n in names):
            raise ValueError("Weight index must refer to files in the checkpoint directory")
    else:
        names = ["model.safetensors"]
    if not names or any(not (source / n).is_file() or (source / n).stat().st_size < 16 for n in names):
        raise ValueError(f"Incomplete safetensors checkpoint: {source}")
    prompt = (ROOT / entry["prompt_file"]).read_text()
    if digest(prompt) != entry["system_prompt_sha256"]:
        raise ValueError("System prompt checksum mismatch")
    native = original_template(source)
    validate_template(native, default_template(native, prompt), prompt, source)
    return names, native, prompt


def validate_template(native, patched, prompt, source):
    from transformers.utils.chat_template_utils import _compile_jinja_template
    tokenizer = read_json(source / "tokenizer_config.json")
    special = {k: v.get("content", "") if isinstance(v, dict) else v
               for k, v in tokenizer.items() if k.endswith("_token")}
    kwargs = dict(special, add_generation_prompt=True, tools=None)
    for content in ["What is shown?", [{"type": "image"}, {"type": "text", "text": "What is shown?"}]]:
        messages = [{"role": "user", "content": content}]
        expected = _compile_jinja_template(native).render(
            messages=[{"role": "system", "content": prompt}] + messages, **kwargs)
        actual = _compile_jinja_template(patched).render(messages=messages, **kwargs)
        if actual != expected:
            raise ValueError("Default system prompt changed native rendering")
        explicit = [{"role": "system", "content": "Custom system prompt."}] + messages
        if _compile_jinja_template(patched).render(messages=explicit, **kwargs) != _compile_jinja_template(native).render(messages=explicit, **kwargs):
            raise ValueError("Explicit system prompt override changed")


def prepare(source, entry, destination, repo_id):
    names, native, prompt = preflight(source, entry)
    destination.mkdir(parents=True, exist_ok=False)
    for name in ASSETS:
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)
    for name in names:
        (destination / name).symlink_to((source / name).resolve())
    if (source / "model.safetensors.index.json").exists():
        shutil.copy2(source / "model.safetensors.index.json", destination)
    for path in source.iterdir():
        if path.is_file() and path.name.upper().startswith(("LICENSE", "NOTICE")):
            shutil.copy2(path, destination / path.name)
    template = default_template(native, prompt)
    (destination / "chat_template.jinja").write_text(template)
    write_json(destination / "chat_template.json", {"chat_template": template})
    tokenizer = read_json(destination / "tokenizer_config.json")
    tokenizer["chat_template"] = template
    # Saved training paths are not part of an inference release.
    tokenizer.pop("name_or_path", None)
    write_json(destination / "tokenizer_config.json", tokenizer)
    config = read_json(destination / "config.json")
    config.pop("_name_or_path", None)
    write_json(destination / "config.json", config)
    processor = destination / "processor_config.json"
    if processor.is_file():
        data = read_json(processor)
        for key, filename in [("image_processor", "preprocessor_config.json"),
                              ("video_processor", "video_preprocessor_config.json")]:
            if key in data and not (destination / filename).exists():
                write_json(destination / filename, data[key])
        if "chat_template" in data:
            data["chat_template"] = template
            write_json(processor, data)
    (destination / "system_prompt.txt").write_text(prompt)
    write_json(destination / "release_provenance.json", {
        "base_model": entry["base_model"], "checkpoint": entry["checkpoint"],
        "selection": "Best checkpoint determined by validation accuracy",
        "system_prompt_sha256": digest(prompt), "native_chat_template_sha256": digest(native),
        "chat_template_sha256": digest(template),
    })
    (destination / "README.md").write_text(
        "---\nlicense: apache-2.0\nlibrary_name: transformers\npipeline_tag: image-text-to-text\n"
        f"base_model: {entry['base_model']}\n---\n\n# {repo_id.split('/')[-1]}\n\n"
        "Checkpoint for *Learning Grounded Reasoning with Human Gaze*, selected by validation accuracy.\n"
        "The condition system prompt is included in the chat template; an explicit system message overrides it.\n"
    )
    return destination


def main():
    release = read_json(ROOT / "configs/release.json")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", default=[], metavar="CONDITION=DIRECTORY",
                        help="Repeat for each selected checkpoint; originals remain unchanged")
    parser.add_argument("--namespace", default=release["namespace"])
    parser.add_argument("--output-dir", type=Path, help="New local staging directory")
    parser.add_argument("--upload", action="store_true", help="Upload the staged models to private repositories")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        for key, entry in release["conditions"].items():
            print(f"{key}: {entry['model_id']} ({entry['checkpoint']})")
        return
    sources = {}
    for value in args.source:
        key, sep, path = value.partition("=")
        if not sep or not path or key not in release["conditions"] or key in sources:
            parser.error("--source must be a unique known CONDITION=DIRECTORY")
        sources[key] = Path(path)
    if not sources:
        parser.error("Provide --source CONDITION=DIRECTORY (see --list)")
    for key, source in sources.items():
        preflight(source, release["conditions"][key])
        print(f"OK {key}")
    if not (args.upload or args.prepare_only):
        return
    output = args.output_dir or Path(tempfile.mkdtemp(prefix="ggr-models-"))
    output.mkdir(parents=True, exist_ok=True)
    prepared = []
    for key, source in sources.items():
        entry = release["conditions"][key]
        repo_id = args.namespace + "/" + entry["model_id"].split("/", 1)[1]
        prepared.append((repo_id, prepare(source, entry, output / key, repo_id)))
    print(f"Prepared {len(prepared)} checkpoint(s) in {output}")
    if not args.upload:
        return
    from huggingface_hub import HfApi
    api = HfApi()
    api.whoami()
    for repo_id, folder in prepared:
        api.create_repo(repo_id, repo_type="model", private=True, exist_ok=args.exist_ok)
        if api.model_info(repo_id).private is not True:
            raise RuntimeError(f"Expected private staging repository: {repo_id}")
        commit = api.upload_folder(repo_id=repo_id, folder_path=str(folder), repo_type="model",
                                   commit_message="Upload validation-selected GGR release checkpoint")
        uploaded = set(api.list_repo_files(repo_id, revision=commit.oid))
        expected = {p.name for p in folder.iterdir() if p.is_file()}
        if expected - uploaded:
            raise RuntimeError(f"Missing uploaded files: {sorted(expected - uploaded)}")
        print(f"Uploaded {repo_id} @ {commit.oid}")


if __name__ == "__main__":
    main()
