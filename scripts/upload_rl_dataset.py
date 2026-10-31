#!/usr/bin/env python3
"""Package the exact RL train/validation rows with embedded images; optionally push."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

EXPECTED = {
    'train': (138651, '51940b4a4f45bbe55869dac78b0f65a7469ef938536313b525a1d8e4669904ed'),
    'validation': (1946, '698cb2c58e6d6cb6ad856ed97c73987170b2acf8c5f6b36ebf54ce59776e8b7a'),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def portable(row):
    """Sanitize only known provenance paths; preserve all task and reward text."""
    from copy import deepcopy
    clean = deepcopy(row)
    metadata = clean.get('extra_info', {}).get('metadata', {})
    for key in ('source_jsonl', 'source_image_id', 'annotation_source',
                'record_path', 'label_path', 'annotated_image_path'):
        value = metadata.get(key)
        if isinstance(value, str) and Path(value).is_absolute():
            metadata[key] = Path(value).name
    return clean


def iter_records(path: Path):
    with path.open() as stream:
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ('data_source', 'ability', 'prompt', 'images', 'reward_model', 'extra_info'):
                if key not in row:
                    raise ValueError(f'{path.name} row {index}: missing {key}')
            if row['ability'] not in ('chart', 'counting', 'search'):
                raise ValueError(f'Unsupported domain at row {index}')
            if 'reward_type' not in row['extra_info'] or 'ground_truth' not in row['reward_model']:
                raise ValueError(f'Missing reward routing at row {index}')
            yield row


def inspect(path: Path, split: str, allow_custom: bool):
    counts = Counter()
    total = 0
    for row in iter_records(path):
        total += 1
        counts[row['ability']] += 1
    digest = sha256(path)
    expected_count, expected_digest = EXPECTED[split]
    if not allow_custom and (total, digest) != (expected_count, expected_digest):
        raise ValueError(f'{split}: input is not the paper snapshot (rows={total}, sha256={digest}); use --allow-custom-data only for a new experiment')
    return {'rows': total, 'source_sha256': digest, 'domains': dict(counts)}


def write_split(path: Path, split: str, output: Path, image_root: Path | None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Features, Image, Sequence, Value

    features = Features({'record': Value('string'), 'images': Sequence(Image(decode=False))})
    schema = features.arrow_schema
    destination = output / 'data'
    destination.mkdir(parents=True, exist_ok=True)
    writer = None
    shard = 0
    shard_bytes = 0
    batch = []
    try:
        for row in iter_records(path):
            images = []
            image_ids = []
            for image in row['images']:
                image_path = Path(image)
                if not image_path.is_file() and image_root is not None:
                    image_path = image_root / (image_path.name if image_path.is_absolute() else image_path)
                payload = image_path.read_bytes()
                digest = hashlib.sha256(payload).hexdigest()
                name = digest + image_path.suffix.lower()
                images.append({'bytes': payload, 'path': name})
                image_ids.append('images/' + name)
                shard_bytes += len(payload)
            clean = portable(row)
            clean['images'] = image_ids
            batch.append({'record': json.dumps(clean, ensure_ascii=False, separators=(',', ':')), 'images': images})
            if len(batch) >= 8:
                if writer is None:
                    writer = pq.ParquetWriter(destination / f'{split}-{shard:05d}.parquet', schema, compression='zstd')
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                batch.clear()
                if shard_bytes >= 1024**3:
                    writer.close(); writer = None; shard += 1; shard_bytes = 0
        if batch:
            if writer is None:
                writer = pq.ParquetWriter(destination / f'{split}-{shard:05d}.parquet', schema, compression='zstd')
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
    finally:
        if writer is not None:
            writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True, help='Original counting-correct-only RL training JSONL')
    parser.add_argument('--validation', type=Path, required=True, help='Original counting-correct-only validation JSONL')
    parser.add_argument('--repo-id', default='gsarch/GGR-RL-Data')
    parser.add_argument('--output', type=Path, default=Path('data/releases/rl'))
    parser.add_argument('--image-root', type=Path, help='Optional root for relocated input images')
    parser.add_argument('--build', action='store_true', help='Build image-embedded Parquet shards locally')
    parser.add_argument('--push', action='store_true', help='Build and upload the dataset')
    parser.add_argument('--public', action='store_true', help='Create a public dataset (private by default)')
    parser.add_argument('--allow-custom-data', action='store_true')
    args = parser.parse_args()
    manifest = {split: inspect(getattr(args, split), split, args.allow_custom_data) for split in EXPECTED}
    print(json.dumps(manifest, indent=2))
    if not (args.build or args.push):
        print('Validated source rows. Pass --build to package or --push to package and upload.')
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit('Output must be empty to prevent stale shards from entering the release')
    args.output.mkdir(parents=True, exist_ok=True)
    for split in EXPECTED:
        write_split(getattr(args, split), split, args.output, args.image_root)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'README.md').write_text('''---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
  - split: validation
    path: data/validation-*.parquet
---
# GGR RL data

Exact RL rows for the GGR and No-Gaze conditions. Images are embedded without
resizing or re-encoding. `record` stores the original JSON fields, including
`ability`, `data_source`, `extra_info.reward_type`, and `reward_model.ground_truth`.
Local source paths are replaced by basenames in provenance metadata.
Use the GGR repository's `scripts/prepare_rl_data.py` to materialize training files.
''')
    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.repo_id, repo_type='dataset', private=not args.public, exist_ok=True)
        api.upload_large_folder(repo_id=args.repo_id, repo_type='dataset', folder_path=str(args.output))
        remote = api.dataset_info(args.repo_id, files_metadata=True)
        remote_sizes = {item.rfilename: item.size for item in remote.siblings}
        expected_files = {str(p.relative_to(args.output)): p.stat().st_size for p in args.output.rglob('*') if p.is_file() and '.cache' not in p.parts}
        missing = [name for name, size in expected_files.items() if remote_sizes.get(name) != size]
        if missing:
            raise RuntimeError('Uploaded file verification failed: ' + ', '.join(missing))
        from huggingface_hub import hf_hub_download
        saved_manifest = Path(hf_hub_download(args.repo_id, 'manifest.json', repo_type='dataset', revision=remote.sha, force_download=True))
        if json.loads(saved_manifest.read_text()) != manifest:
            raise RuntimeError('Remote manifest differs from packaged data')
        print(f'Verified {len(expected_files)} files and split counts at revision {remote.sha}')
        print(f'https://huggingface.co/datasets/{args.repo_id}')


if __name__ == '__main__':
    main()
