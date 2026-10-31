#!/usr/bin/env python3
"""Download and register GGR/No-Gaze SFT data, or materialize a regenerated package."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
RELEASE = json.loads((ROOT / 'configs/release.json').read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def safe_relative(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Nonportable dataset path: {value}')
    return path


def materialize(package, source_file, destination, media_dir, condition):
    rows = json.loads(source_file.read_text())
    prompt = (ROOT / 'prompts' / 'ggr.txt').read_text().strip()
    copied = set()
    for row in rows:
        row['messages'][0]['content'] = prompt
        images = []
        for value in row['images']:
            relative = safe_relative(value)
            source = package / relative
            target = media_dir / relative.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if relative not in copied:
                if not target.exists():
                    shutil.copyfile(source, target)
                elif digest(source) != digest(target):
                    raise ValueError(f'Image collision: {relative.name}')
                copied.add(relative)
            images.append(os.path.relpath(target, ROOT))
        row['images'] = images
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(destination)
    return len(rows)


def register(registry, name, file):
    data = json.loads(registry.read_text()) if registry.exists() else {}
    record = {'file_name': os.path.relpath(file, registry.parent), 'formatting': 'sharegpt',
              'columns': {'messages': 'messages', 'images': 'images'},
              'tags': {'role_tag': 'role', 'content_tag': 'content', 'user_tag': 'user',
                       'assistant_tag': 'assistant', 'system_tag': 'system'}}
    if name in data and data[name] != record:
        raise ValueError(f'Registry already contains a different {name} entry')
    data[name] = record
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(json.dumps(data, indent=2) + '\n')


def image_reference(value, repo_id):
    """Resolve a dataset image URL within its pinned Hub repository."""
    parsed = urlparse(value)
    if parsed.scheme:
        prefix = f'/datasets/{repo_id}/resolve/'
        if parsed.scheme != 'https' or parsed.netloc != 'huggingface.co' or not parsed.path.startswith(prefix):
            raise ValueError('Image URL does not belong to the selected dataset')
        _, separator, relative = parsed.path[len(prefix):].partition('/')
        if not separator:
            raise ValueError('Image URL has no repository-relative filename')
        value = unquote(relative)
    path = safe_relative(value)
    if not path.parts or path.parts[0] != 'images':
        raise ValueError('Expected an images/ asset')
    return path.as_posix()


def restore_parquet_row(example, repo_id, revision, media_dir, fetch):
    row = json.loads(example.get('extra_fields_json') or '{}')
    row.update(id=example['id'], messages=[dict(m) for m in example['messages']],
               gt_answer=example['gt_answer'], metadata=json.loads(example.get('metadata_json') or '{}'))
    if [m['role'] for m in row['messages']] != ['system', 'user', 'assistant']:
        raise ValueError('Unexpected SFT message roles')
    row['messages'][0]['content'] = (ROOT / 'prompts/ggr.txt').read_text().strip()
    row['images'] = []
    media_dir.mkdir(parents=True, exist_ok=True)
    for image in example['images']:
        relative = image_reference(image['path'], repo_id)
        target = media_dir / Path(relative).name
        if not target.exists():
            if image.get('bytes') is not None:
                payload = image['bytes']
                if hashlib.sha256(payload).hexdigest() != target.stem:
                    raise ValueError('Embedded image checksum mismatch')
                target.write_bytes(payload)
            else:
                source = Path(fetch(repo_id, relative, repo_type='dataset', revision=revision))
                if digest(source) != target.stem:
                    raise ValueError('Downloaded image checksum mismatch')
                shutil.copyfile(source, target)
        elif digest(target) != target.stem:
            raise ValueError('Local image checksum mismatch')
        row['images'].append(os.path.relpath(target, ROOT))
    if row['messages'][1]['content'].count('<image>') != len(row['images']):
        raise ValueError('Image placeholder count mismatch')
    return row


def prepare_parquet(args, conditions, revision):
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download, hf_hub_download
    configurations = RELEASE['datasets']['sft']['configurations']
    expected = RELEASE['datasets']['sft']['rows']
    for condition in conditions:
        config = configurations[condition]
        package = Path(snapshot_download(args.repo_id, repo_type='dataset', revision=revision,
                                        allow_patterns=[f'data/{config}/train-*.parquet']))
        shards = sorted((package / 'data' / config).glob('train-*.parquet'))
        count = sum(pq.ParquetFile(p).metadata.num_rows for p in shards)
        if not shards or (args.repo_id == RELEASE['datasets']['sft']['repo_id'] and count != expected[condition]):
            raise ValueError(f'{condition}: unexpected uploaded row count {count}')
        destination = args.output / condition / 'train.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.tmp')
        seen = set()
        with temporary.open('w') as handle:
            handle.write('[\n')
            index = 0
            for shard in shards:
                for batch in pq.ParquetFile(shard).iter_batches(batch_size=128):
                    for example in batch.to_pylist():
                        row = restore_parquet_row(example, args.repo_id, revision, args.output / 'images', hf_hub_download)
                        if row['id'] in seen:
                            raise ValueError('Duplicate SFT row ID')
                        seen.add(row['id'])
                        if index:
                            handle.write(',\n')
                        json.dump(row, handle, ensure_ascii=False)
                        index += 1
            handle.write('\n]\n')
        temporary.replace(destination)
        if args.register:
            register(args.register, condition, destination)
        print(f'{condition}: {count} rows -> {os.path.relpath(destination, ROOT)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=['ggr', 'no_gaze', 'validation', 'all'], default='all')
    parser.add_argument('--repo-id', default=RELEASE['datasets']['sft']['repo_id'])
    parser.add_argument('--revision', help='Defaults to the pinned release revision')
    parser.add_argument('--validation-repo', default=RELEASE['datasets']['sft_validation']['repo_id'])
    parser.add_argument('--package', type=Path, help='Use a local regenerated package instead of downloading')
    parser.add_argument('--output', type=Path, default=ROOT / 'data' / 'sft')
    parser.add_argument('--register', nargs='?', const=ROOT / 'ggr-sft' / 'data' / 'dataset_info.json', type=Path)
    args = parser.parse_args()
    conditions = ['ggr', 'no_gaze', 'validation'] if args.condition == 'all' else [args.condition]
    if args.package:
        if len(conditions) != 1 or args.condition == 'validation':
            parser.error('--package requires --condition ggr or no_gaze')
        package = args.package
        manifest = None
    else:
        from huggingface_hub import HfApi, snapshot_download
        api = HfApi()
        revision = args.revision or (RELEASE['datasets']['sft'].get('revision')
                                    if args.repo_id == RELEASE['datasets']['sft']['repo_id'] else None)
        info = api.dataset_info(args.repo_id, revision=revision)
        revision = info.sha
        if 'manifest.json' not in {f.rfilename for f in info.siblings}:
            validation_package = None
            if 'validation' in conditions:
                try:
                    validation_package = Path(snapshot_download(args.validation_repo, repo_type='dataset',
                        allow_patterns=['manifest.json', 'validation/*.json', 'images/*']))
                except Exception as error:
                    raise SystemExit('The SFT validation package is not available. Upload it using '
                        'scripts/upload_sft_dataset.py --condition validation --repo-id ' + args.validation_repo +
                        ' (see docs/DATA.md). To prepare training only, use --condition ggr or --condition no_gaze.') from error
                validation_manifest = json.loads((validation_package / 'manifest.json').read_text())
                spec = validation_manifest['configurations']['validation']
                source = validation_package / safe_relative(spec['file'])
                if digest(source) != spec['sha256'] or len(json.loads(source.read_text())) != 100:
                    raise ValueError('Invalid SFT validation snapshot')
            prepare_parquet(args, [c for c in conditions if c != 'validation'], revision)
            if validation_package is not None:
                destination = args.output / 'validation/validation.json'
                materialize(validation_package, source, destination, args.output / 'images', 'validation')
                if args.register:
                    register(args.register, 'ggr_validation', destination)
                print('validation: 100 rows')
            return
        try:
            package = Path(snapshot_download(args.repo_id, repo_type='dataset', revision=revision,
                                            allow_patterns=['manifest.json'] + [f'{c}/*.json' for c in conditions]))
        except Exception as error:
            raise SystemExit(f'Cannot download {args.repo_id}: {error}\nPublish the accepted snapshots with scripts/upload_sft_dataset.py first.') from error
        resolved_revision = package.name
        manifest = json.loads((package / 'manifest.json').read_text())
        required_images = set()
        for condition in conditions:
            spec = manifest['configurations'][condition]
            source = package / safe_relative(spec['file'])
            if digest(source) != spec['sha256']:
                raise ValueError(f'Corrupt downloaded {condition} data')
            rows = json.loads(source.read_text())
            if len(rows) != spec['rows']:
                raise ValueError(f'Incorrect {condition} row count')
            required_images.update(str(safe_relative(p)) for row in rows for p in row['images'])
        snapshot_download(args.repo_id, repo_type='dataset', revision=resolved_revision,
                          allow_patterns=sorted(required_images))
    for condition in conditions:
        filename = 'validation.json' if condition == 'validation' else 'train.json'
        source = package / (manifest['configurations'][condition]['file'] if manifest else filename)
        destination = args.output / condition / filename
        count = materialize(package, source, destination, args.output / 'images', condition)
        if args.register:
            register(args.register, 'ggr_validation' if condition == 'validation' else condition, destination)
        print(f'{condition}: {count} rows -> {os.path.relpath(destination, ROOT)}')


if __name__ == '__main__':
    main()
