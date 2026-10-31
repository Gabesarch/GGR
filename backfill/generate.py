#!/usr/bin/env python3
"""Generate single-stage GGR or No-Gaze SFT annotations from human inputs."""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backfill.normalize_sft_point_text import normalize_text

PROMPTS = Path(__file__).resolve().parent / 'prompts'
RAW_REPO = 'gsarch/backfill-stage1-human-inputs'
RAW_REVISION = 'f87f7416928b048ef76ade6ab22a5705a08e22fd'
ERROR_FLAGS = ('major_visual_error', 'countable_unit_or_double_count_error',
               'relation_geometry_error', 'ocr_error', 'chart_or_table_error',
               'dynamic_claim_error', 'missing_coordinate_grounding',
               'self_contradictory', 'severe_looping_or_irrelevance')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def category(row):
    cohort = row['cohort']
    if cohort in ('new_counting', 'new_reviewed_counting'):
        return 'counting'
    if cohort == 'new_search':
        return 'search'
    if cohort.startswith('old_') and cohort[4:] in ('chart', 'counting', 'search'):
        return cohort[4:]
    raise ValueError(f'Unknown source cohort: {cohort}')


def teacher_prompt(row, condition):
    text = f"Question: {row['question']} Ground your reasoning in (x, y) coordinates."
    if condition == 'ggr':
        points = row['fixation_list']
        if not points:
            raise ValueError('GGR requires human fixation_list')
        for point in points:
            if (len(point) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                      or not math.isfinite(x) for x in point)):
                raise ValueError(f'Invalid fixation point: {point}')
        # Coordinates already are normalized to 0-1000 by the published source.
        # Preserve order, duplicates, and out-of-bounds values without clipping.
        text += '\n\nhuman-attended regions:\n\n' + '\n'.join(
            f'- ({x:g}, {y:g})' for x, y in points)
    return text


def parse_judgment(text):
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip(), flags=re.I)
    result = json.loads(text)
    keys = ('aligned', 'visually_correct', *ERROR_FLAGS)
    if not isinstance(result, dict) or any(type(result.get(k)) is not bool for k in keys):
        raise ValueError('Judge did not return all required boolean fields')
    if type(result.get('quality_score')) is not int or not 1 <= result['quality_score'] <= 5:
        raise ValueError('Invalid judge quality_score')
    if not isinstance(result.get('reason'), str):
        raise ValueError('Missing judge reason')
    return result


def passes(judgment):
    return (judgment['aligned'] and judgment['visually_correct']
            and not any(judgment[k] for k in ERROR_FLAGS))


def image_data(image):
    from qwen_vl_utils import fetch_image
    image = fetch_image({'image': image, 'min_pixels': 2_000_000, 'max_pixels': 10_000_000})
    buffer = io.BytesIO()
    image.convert('RGB').save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


def messages(system, text, image_url=None):
    content = [{'type': 'text', 'text': text}]
    if image_url:
        content.insert(0, {'type': 'image_url', 'image_url': {'url': image_url}})
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}]


def call_model(client, args, model, request_messages, kind, n=1):
    generation = kind == 'teacher'
    extra = {'chat_template_kwargs': {'enable_thinking': False}}
    extra.update(top_k=40 if generation else 20, min_p=0, repetition_penalty=1.05)
    response = client.chat.completions.create(
        model=model, messages=request_messages, n=n,
        temperature=0.95 if generation else 0.6,
        max_tokens=16000 if generation else 8000,
        top_p=0.95, presence_penalty=0, seed=20260825, extra_body=extra,
    )
    if len(response.choices) != n:
        raise ValueError(f'Expected {n} completions, received {len(response.choices)}')
    for choice in response.choices:
        if choice.finish_reason not in ('stop', 'eos') or not choice.message.content:
            raise ValueError(f'Incomplete {kind} completion: {choice.finish_reason}')
    texts = [choice.message.content for choice in response.choices]
    return texts if generation else texts[0]


def process_trial(row, args):
    key = hashlib.sha256(row['sample_id'].encode()).hexdigest()[:24]
    target = args.output / 'trials' / f'{key}.json'
    saved = json.loads(target.read_text()) if target.exists() else {
        'sample_id': row['sample_id'], 'condition': args.condition,
        'category': category(row), 'cohort': row['cohort'], 'question': row['question'],
        'ground_truth_answer': row['answer'], 'image_pixel_sha256': row['image_pixel_sha256'],
        'samples': [], 'complete': False,
    }
    if saved.get('complete'):
        return len(saved['samples']), sum(s.get('accepted', False) for s in saved['samples'])
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=os.environ.get(args.api_key_env, 'EMPTY'),
                    max_retries=3, timeout=1800)
    judge_client = OpenAI(base_url=args.judge_base_url, api_key=os.environ.get(args.api_key_env, 'EMPTY'),
                          max_retries=3, timeout=300)
    url = image_data(row['image'])
    system = (PROMPTS / args.condition / f'{category(row)}.txt').read_text().strip()
    judge_prompt = (PROMPTS / 'judge.txt').read_text().strip()
    slots = {s['sample_idx']: s for s in saved['samples']}
    missing = [i for i in range(args.samples) if 'text' not in slots.get(i, {})]
    if missing:
        try:
            texts = call_model(client, args, args.teacher_model,
                messages(system, teacher_prompt(row, args.condition), url), 'teacher', n=len(missing))
            for index, text in zip(missing, texts):
                slots[index] = {'sample_idx': index, 'text': text}
            saved['samples'] = [slots[k] for k in sorted(slots)]
            write_json(target, saved)
        except Exception as error:
            saved['error'] = f'{type(error).__name__}: {error}'
            write_json(target, saved)
            return 0, 0
    for index in range(args.samples):
        sample = slots.setdefault(index, {'sample_idx': index})
        if sample.get('status') == 'complete':
            continue
        try:
            candidate = sample['text']
            canonical, _ = normalize_text(candidate)
            prompt = (judge_prompt.replace('{question}', row['question'])
                      .replace('{reference_answer}', row['answer'])
                      .replace('{candidate_response}', canonical))
            raw = call_model(judge_client, args, args.judge_model,
                             messages('Audit answer alignment and visual correctness. Return only the requested JSON.', prompt, url), 'judge')
            sample['judgment'] = parse_judgment(raw)
            sample['accepted'] = passes(sample['judgment'])
            sample.update(status='complete', thinking_closed=None)
            sample.pop('error', None)
        except Exception as error:
            # Preserve successful candidates and retry only failed slots on rerun.
            sample.update(status='error', error=f'{type(error).__name__}: {error}')
        saved['samples'] = [slots[k] for k in sorted(slots)]
        write_json(target, saved)
    saved['complete'] = (len(saved['samples']) == args.samples
                         and all(s.get('status') == 'complete' for s in saved['samples']))
    write_json(target, saved)
    return sum(s.get('status') == 'complete' for s in saved['samples']), sum(s.get('accepted', False) for s in saved['samples'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=['ggr', 'no_gaze'], required=True)
    parser.add_argument('--dataset', default=RAW_REPO)
    parser.add_argument('--revision', default=RAW_REVISION)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--base-url', default='http://localhost:8000/v1', help='Teacher vLLM server URL')
    parser.add_argument('--judge-base-url', help='Judge server URL; defaults to the teacher server')
    parser.add_argument('--api-key-env', default='VLLM_API_KEY')
    parser.add_argument('--teacher-model')
    parser.add_argument('--judge-model')
    parser.add_argument('--samples', type=int, default=25)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--dry-run', action='store_true', help='Inspect requests without model calls or output writes')
    args = parser.parse_args()
    if args.samples < 1 or args.workers < 1 or args.limit is not None and args.limit < 1:
        parser.error('samples, workers and limit must be positive')
    args.output = args.output or Path('data/backfill') / args.condition
    args.teacher_model = args.teacher_model or 'Qwen/Qwen3.5-122B-A10B'
    args.judge_model = args.judge_model or args.teacher_model
    args.judge_base_url = args.judge_base_url or args.base_url
    from datasets import load_dataset
    data = load_dataset(args.dataset, revision=args.revision)
    # Check published decoded-pixel identities, regardless of question/answer.
    def keys(split):
        return set(zip(split['image_width'], split['image_height'], split['image_pixel_sha256']))
    if keys(data['train']) & keys(data['validation']):
        raise ValueError('Training contains a held-out validation image')
    train = data['train']
    if args.limit:
        train = train.select(range(min(args.limit, len(train))))
    if args.dry_run:
        row = train[0]
        print(json.dumps({'condition': args.condition, 'trials': len(train), 'samples_per_trial': args.samples,
                          'teacher_model': args.teacher_model, 'judge_model': args.judge_model,
                          'system': (PROMPTS / args.condition / f'{category(row)}.txt').read_text(),
                          'user': teacher_prompt(row, args.condition)}, indent=2))
        return
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in ('workers', 'dry_run', 'output', 'api_key_env')}
    manifest.update(source_kind='reviewed_human_inputs_regeneration', expected_trials=len(train),
                    prompts_sha256={p.relative_to(PROMPTS).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in PROMPTS.rglob('*.txt')})
    path = args.output / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Existing run configuration differs; use another --output directory')
    write_json(path, manifest)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        counts = list(pool.map(lambda row: process_trial(row, args), train))
    summary = {'trials': len(counts), 'requested_candidates': len(counts) * args.samples,
               'completed_candidates': sum(n for n, _ in counts), 'accepted_candidates': sum(n for _, n in counts)}
    summary['complete'] = summary['completed_candidates'] == summary['requested_candidates']
    write_json(args.output / 'summary.json', summary)
    print(json.dumps(summary, indent=2))
    if not summary['complete']:
        raise SystemExit('Some requests failed; rerun the same command to retry them.')


if __name__ == '__main__':
    main()
