#!/usr/bin/env python3
"""Download GGR RL data and restore portable VERL JSONL plus image files."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def restore_row(example, output: Path):
    record = json.loads(example['record'])
    references = record['images']
    if len(references) != len(example['images']):
        raise ValueError('Image count does not match record')
    for reference, image in zip(references, example['images'], strict=True):
        relative = Path(reference)
        if relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.parts[0] != 'images':
            raise ValueError('Dataset contains an unsafe image reference')
        payload = image.get('bytes')
        if not payload:
            raise ValueError('Dataset must contain embedded image bytes')
        if hashlib.sha256(payload).hexdigest() != relative.stem:
            raise ValueError('Embedded image checksum does not match its filename')
        path = output / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != relative.stem:
            raise ValueError(f'Local image checksum mismatch: {relative}')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', default='gsarch/GGR-RL-Data')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--output', type=Path, default=Path('data/rl'))
    args = parser.parse_args()
    from datasets import Image, Sequence, load_dataset
    from huggingface_hub import hf_hub_download
    manifest = json.loads(Path(hf_hub_download(args.repo_id, 'manifest.json', repo_type='dataset', revision=args.revision)).read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ('train', 'validation'):
        dataset = load_dataset(args.repo_id, split=split, revision=args.revision, streaming=True)
        dataset = dataset.cast_column('images', Sequence(Image(decode=False)))
        temporary = args.output / f'{split}.jsonl.partial'
        counts = Counter()
        total = 0
        with temporary.open('w') as stream:
            for example in dataset:
                row = restore_row(example, args.output)
                counts[row['ability']] += 1
                total += 1
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        if total != manifest[split]['rows'] or dict(counts) != manifest[split]['domains']:
            raise ValueError(f'{split}: rows or routing counts differ from uploaded manifest')
        temporary.replace(args.output / f'{split}.jsonl')
        summary[split] = {'rows': total, 'domains': dict(counts)}
    (args.output / 'manifest.json').write_text(json.dumps({'repo_id': args.repo_id, 'revision': args.revision, 'splits': summary}, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
