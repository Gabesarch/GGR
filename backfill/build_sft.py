#!/usr/bin/env python3
"""Format complete single-stage annotations as a portable SFT dataset."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backfill.assistant_language_cleanup import clean_assistant_language, remaining_language_artifacts
from backfill.formatting import (assistant_think_tag_structure, balance_unique_samples_by_category,
    format_assistant_content, format_user_content, key_digest, parse_count_target, question_image_key)
from backfill.generate import write_json
from backfill.normalize_sft_point_text import normalize_text, remaining_noncanonical, PAREN_NUMERIC_POINT_RE


def format_candidate(trial, sample, system):
    if (sample.get('status') != 'complete' or 'thinking_closed' not in sample
            or sample['thinking_closed'] is not None):
        return None
    text = format_assistant_content(sample['text'])
    if not text or not assistant_think_tag_structure(text)[0]:
        return None
    identity = f"{trial['sample_id']}:{sample['sample_idx']}"
    text, _ = clean_assistant_language(text, variation_key=identity)
    text, _ = normalize_text(text)
    text, _ = clean_assistant_language(text, variation_key=identity)
    if remaining_language_artifacts(text) or remaining_noncanonical(text):
        raise ValueError(f'Residual annotation language or noncanonical points: {identity}')
    if not PAREN_NUMERIC_POINT_RE.search(text) or not assistant_think_tag_structure(text)[0]:
        return None
    question, _ = normalize_text(format_user_content(trial['question']))
    system, _ = normalize_text(system)
    return {'id': identity, 'messages': [{'role': 'system', 'content': system},
        {'role': 'user', 'content': question}, {'role': 'assistant', 'content': text}],
        'images': [], 'gt_answer': trial['ground_truth_answer'],
        'metadata': {'sample_id': trial['sample_id'], 'sample_idx': sample['sample_idx'],
            'cohort': trial['cohort'], 'category': trial['category'],
            'count_target': parse_count_target(trial['ground_truth_answer']) if trial['category'] == 'counting' else None,
            'question_image_key_sha256': key_digest(question_image_key(trial['image_pixel_sha256'], trial['question'])),
            'accepted': sample['accepted'], 'condition': trial['condition']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New package directory')
    parser.add_argument('--system-prompt', type=Path)
    parser.add_argument('--seed', type=int, default=20260830)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new --output directory; existing datasets are not overwritten')
    manifest = json.loads((args.run_dir / 'manifest.json').read_text())
    summary = json.loads((args.run_dir / 'summary.json').read_text())
    if not summary.get('complete'):
        raise ValueError('Finish all generation/judging before SFT preparation')
    condition = manifest['condition']
    prompt = args.system_prompt or ROOT / 'prompts' / 'ggr.txt'
    system = prompt.read_text().strip()
    rows = []
    counter = Counter()
    files = sorted((args.run_dir / 'trials').glob('*.json'))
    if len(files) != manifest['expected_trials']:
        raise ValueError('Trial count differs from the run manifest')
    for path in files:
        trial = json.loads(path.read_text())
        if not trial['complete'] or len(trial['samples']) != manifest['samples']:
            raise ValueError(f'Incomplete trial: {path.name}')
        for sample in trial['samples']:
            counter['generated'] += 1
            if not sample['accepted']:
                counter['judge_rejected'] += 1
                continue
            row = format_candidate(trial, sample, system)
            if row is None:
                counter['format_rejected'] += 1
            else:
                rows.append(row)
    # Original cohorts are balanced before new counting/search are appended.
    old = [r for r in rows if r['metadata']['cohort'].startswith('old_')]
    appended = [r for r in rows if not r['metadata']['cohort'].startswith('old_')]
    old, balance_audit = balance_unique_samples_by_category(
        old, seed=args.seed, max_samples_per_question=25, max_category_ratio=1.25)
    rows = old + appended
    from datasets import Image as HFImage, load_dataset
    source = load_dataset(manifest['dataset'], revision=manifest['revision'], split='train').cast_column('image', HFImage(decode=False))
    index = {key: i for i, key in enumerate(source['sample_id'])}
    image_paths = {}
    args.output.mkdir(parents=True)
    for row in rows:
        key = row['metadata']['sample_id']
        if key not in image_paths:
            value = source[index[key]]['image']
            encoded = value['bytes'] if value['bytes'] is not None else Path(value['path']).read_bytes()
            from PIL import Image
            with Image.open(io.BytesIO(encoded)) as image:
                suffix = (image.format or 'png').lower().replace('jpeg', 'jpg')
            relative = f'images/{hashlib.sha256(encoded).hexdigest()}.{suffix}'
            destination = args.output / relative
            destination.parent.mkdir(exist_ok=True)
            if not destination.exists():
                destination.write_bytes(encoded)
            image_paths[key] = relative
        row['images'] = [image_paths[key]]
    write_json(args.output / 'train.json', rows)
    write_json(args.output / 'audit.json', {'kind': 'regenerated_from_reviewed_human_inputs',
        'condition': condition, 'dataset': manifest['dataset'], 'revision': manifest['revision'],
        'rows': len(rows), 'counters': dict(counter), 'old_cohort_balance': balance_audit,
        'cohort_counts': dict(Counter(r['metadata']['cohort'] for r in rows)),
        'system_prompt_sha256': hashlib.sha256(prompt.read_bytes()).hexdigest(),
        'train_sha256': hashlib.sha256((args.output / 'train.json').read_bytes()).hexdigest()})
    print(f'Prepared {len(rows)} regenerated {condition} rows in {args.output}')


if __name__ == '__main__':
    main()
