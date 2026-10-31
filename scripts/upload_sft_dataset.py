#!/usr/bin/env python3
"""Validate and optionally upload the paper's accepted SFT data snapshots."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'backfill' / 'paper_data.json'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_metadata(value):
    """Keep useful row provenance while removing machine-specific paths."""
    if isinstance(value, dict):
        return {k: safe_metadata(v) for k, v in value.items()
                if not any(term in k for term in ('path', 'directory', 'root'))
                and not (isinstance(v, str) and Path(v).is_absolute())}
    if isinstance(value, list):
        return [safe_metadata(v) for v in value]
    return value


def validate_row(row):
    messages = row['messages']
    if [m['role'] for m in messages] != ['system', 'user', 'assistant']:
        raise ValueError(f'Invalid message roles: {row.get("id")}')
    if any(set(m) != {'role', 'content'} or not isinstance(m['content'], str) for m in messages):
        raise ValueError('Unsupported message schema')
    if messages[1]['content'].count('<image>') != len(row['images']) or len(row['images']) != 1:
        raise ValueError('Expected one image and one placeholder')
    if not isinstance(row.get('gt_answer'), str):
        raise ValueError('Original gt_answer must be a string')


def package_condition(name, spec, source_root, image_root, output, image_cache, source=None):
    source = source if source is not None else source_root / spec['source_file']
    if sha256(source) != spec['source_sha256']:
        raise ValueError(f'{name}: source does not match the paper manifest')
    rows = json.loads(source.read_text())
    if len(rows) != spec['rows']:
        raise ValueError(f'{name}: expected {spec["rows"]} rows, found {len(rows)}')
    ids = set()
    counts = Counter()
    folder = output / name
    folder.mkdir(parents=True, exist_ok=False)
    target = folder / ('validation.json' if name == 'validation' else 'train.json')
    with target.open('w') as handle:
        handle.write('[\n')
        for index, row in enumerate(rows):
            validate_row(row)
            if not isinstance(row['id'], str) or not row['id'] or row['id'] in ids:
                raise ValueError(f'{name}: invalid or duplicate ID {row.get("id")}')
            ids.add(row['id'])
            images = []
            for raw in row['images']:
                path = Path(raw)
                path = (path if path.is_absolute() else image_root / path).resolve(strict=True)
                if path not in image_cache:
                    digest = sha256(path)
                    relative = f'images/{digest}{path.suffix.lower()}'
                    destination = output / relative
                    destination.parent.mkdir(exist_ok=True)
                    if not destination.exists():
                        shutil.copyfile(path, destination)
                    image_cache[path] = relative
                images.append(image_cache[path])
            row['messages'][0]['content'] = (ROOT / 'prompts' / 'ggr.txt').read_text().strip()
            clean = {'id': row['id'], 'messages': row['messages'], 'images': images,
                     'gt_answer': row['gt_answer'], 'metadata': safe_metadata(row.get('metadata', {}))}
            if 'category' in clean['metadata']:
                counts[clean['metadata']['category']] += 1
            if index:
                handle.write(',\n')
            json.dump(clean, handle, ensure_ascii=False)
        handle.write('\n]\n')
    if sha256(source) != spec['source_sha256']:
        raise ValueError(f'{name}: source changed while packaging')
    return {'rows': len(rows), 'source_sha256': spec['source_sha256'],
            'file': target.relative_to(output).as_posix(), 'sha256': sha256(target),
            'category_counts': dict(counts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path('.'), help='Root containing the canonical SFT snapshots')
    parser.add_argument('--source', action='append', default=[], metavar='CONDITION=FILE',
                        help='Override a ggr, no_gaze, or validation source snapshot; repeat as needed')
    parser.add_argument('--image-root', type=Path, help='Root for relative image paths in original data')
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--output', type=Path, default=Path('data/upload/sft'))
    parser.add_argument('--repo-id', default='gsarch/GGR-SFT-Data')
    parser.add_argument('--condition', nargs='+', choices=['ggr', 'no_gaze', 'validation'])
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--push', action='store_true')
    parser.add_argument('--public', action='store_true', help='Create a public repository (default private)')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    selected = args.condition or list(manifest['conditions'])
    sources = {}
    for value in args.source:
        name, separator, filename = value.partition('=')
        if not separator or not filename or name not in selected or name in sources:
            parser.error('--source must be a unique selected CONDITION=FILE')
        sources[name] = Path(filename)
    for name in selected:
        spec = manifest['conditions'][name]
        source = sources.get(name, args.source_root / spec['source_file'])
        actual = sha256(source)
        if actual != spec['source_sha256']:
            raise ValueError(f'{name}: content hash changed; refusing a mislabeled paper snapshot')
        print(f'{name}: source SHA256 verified; expected rows {spec["rows"]}')
    if not (args.prepare_only or args.push):
        print('Source files verified. Pass --prepare-only or --push to package images and rows.')
        return
    if args.image_root is None:
        parser.error('--prepare-only and --push require --image-root')
    if args.output.exists():
        parser.error('Choose a new --output directory; existing packages are not overwritten')
    args.output.mkdir(parents=True)
    image_cache = {}
    configurations = {name: package_condition(name, manifest['conditions'][name], args.source_root,
        args.image_root, args.output, image_cache, source=sources.get(name)) for name in selected}
    package = {'schema_version': 1, 'kind': 'paper_accepted_snapshots', 'configurations': configurations,
               'upstream_generation_complete': False,
               'note': 'These are the complete selected training files. Their upstream generation/judging pools were partial snapshots.',
               'portable_transform': 'Shared SFT system prompt; user/assistant messages, original answers, IDs and order unchanged; image paths made relative; machine paths removed from metadata.'}
    (args.output / 'manifest.json').write_text(json.dumps(package, indent=2) + '\n')
    card = ('---\ntask_categories:\n- image-text-to-text\n---\n\n# GGR SFT data\n\n'
            'Accepted paper training snapshots for GGR and No-Gaze, with the common SFT validation set. '
            'Rows preserve original user/assistant messages, answers and order. Images are stored once under `images/`. '
            'Download and register with `scripts/prepare_sft_data.py` from the GGR code release.\n\n'
            'The selected training files are complete snapshots; upstream generation and judging did not finish every possible candidate. '
            'Fresh backfill from the reviewed human-input dataset produces a new dataset.\n\n'
            'Image licenses and dataset terms remain those of the original sources.\n')
    (args.output / 'README.md').write_text(card)
    print(json.dumps(package, indent=2))
    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        if api.repo_exists(args.repo_id, repo_type='dataset'):
            raise ValueError('Destination already exists; choose a new --repo-id to avoid replacing a release')
        api.create_repo(args.repo_id, repo_type='dataset', private=not args.public)
        result = api.upload_folder(repo_id=args.repo_id, repo_type='dataset', folder_path=args.output,
                                   commit_message='Release accepted GGR and No-Gaze SFT snapshots')
        files = set(api.list_repo_files(args.repo_id, repo_type='dataset', revision=result.oid))
        expected = {p.relative_to(args.output).as_posix() for p in args.output.rglob('*') if p.is_file()}
        if not expected <= files:
            raise RuntimeError('Remote verification found missing files')
        print(f'Uploaded https://huggingface.co/datasets/{args.repo_id} at {result.oid}')


if __name__ == '__main__':
    main()
