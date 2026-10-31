#!/usr/bin/env python3
"""Run the paper's six benchmarks and condition-fitted gaze alignment."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

from ggr_eval.tasks import TASKS, load_task, module_for
from ggr_eval.generation import RESPONSE_TOKENS, sampling_kwargs

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('no-gaze-sft', 'ggr-sft', 'no-gaze-rl', 'ggr-rl')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--condition', choices=CONDITIONS, required=True)
    p.add_argument('--model', help='Override the Hugging Face model ID or evaluate a local trained checkpoint')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tasks', default=','.join(TASKS), help='Comma-separated task names, or all')
    p.add_argument('--repeats', type=int, default=3, help='Independent benchmark runs; gaze has 3 rollouts per run')
    p.add_argument('--tensor-parallel-size', type=int, default=1)
    p.add_argument('--judge-tensor-parallel-size', type=int, default=1)
    p.add_argument('--gpu-memory-utilization', type=float, default=.9)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--limit', type=int, help='Smoke-test only; marks results partial')
    p.add_argument('--stage', choices=('all','generate','score'), default='all')
    p.add_argument('--repeat-index', type=int, help=argparse.SUPPRESS)
    p.add_argument('--dry-run', action='store_true')
    return p


def configuration(args):
    config = json.loads((ROOT / 'configs/release.json').read_text())['conditions'][args.condition].copy()
    if args.model:
        config['model_id'] = args.model
        config.pop('revision', None)
    prompt = (ROOT / config['prompt_file']).read_text()
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    if config.get('system_prompt_sha256') and config['system_prompt_sha256'] != digest:
        raise ValueError('System prompt hash does not match the release manifest')
    config['system_prompt_sha256'] = digest
    return config, prompt


def selected_tasks(args):
    tasks = list(TASKS) if args.tasks == 'all' else args.tasks.split(',')
    if not tasks or any(t not in TASKS for t in tasks) or len(set(tasks)) != len(tasks):
        raise ValueError('Tasks must be a unique subset of: ' + ','.join(TASKS))
    return tasks


def sample_path(args, task, repeat):
    return args.output / f'{task}.run{repeat + 1}.jsonl'


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def generate(args, config, prompt, tasks):
    from vllm import LLM, SamplingParams
    from ggr_eval.generation import make_messages, actor_kwargs, sampling_kwargs
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    engine_args = actor_kwargs(config['model_id'], args.tensor_parallel_size, args.gpu_memory_utilization,
                               args.seed + args.repeat_index)
    if config.get('revision'):
        engine_args['revision'] = config['revision']
    engine = LLM(**engine_args)
    for task in tasks:
        target = sample_path(args, task, args.repeat_index)
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite {target}; use a fresh output directory')
        spec, module = TASKS[task], module_for(task)
        dataset = load_task(task, args.limit)
        # Source held_out_gaze evaluates three generated responses per trial.
        rollouts = 3 if task == 'gaze' else 1
        params = SamplingParams(**sampling_kwargs(spec['tokens']))
        with target.open('x') as handle:
            for start in range(0, len(dataset), args.batch_size):
                docs = [dataset[i] for i in range(start, min(start + args.batch_size, len(dataset)))]
                chats = []
                for doc in docs:
                    question = getattr(module, spec['text'])(doc, {k: spec.get(k, '') for k in ('pre_prompt','post_prompt')})
                    images = getattr(module, spec['visual'])(doc)
                    chat = make_messages(images, question, prompt, config['max_image_pixels'], config['min_resize_pixels'])
                    chats.extend([chat] * rollouts)
                outputs = engine.chat(messages=chats, sampling_params=params, use_tqdm=True)
                if len(outputs) != len(docs) * rollouts:
                    raise RuntimeError('Generation returned incomplete coverage')
                for offset, doc in enumerate(docs):
                    responses = []
                    for result in outputs[offset * rollouts:(offset + 1) * rollouts]:
                        if len(result.outputs) != 1:
                            raise RuntimeError('Expected exactly one completion per request')
                        responses.append(result.outputs[0].text)
                    clean_doc = {k: v for k, v in doc.items() if k not in ('image','images','Image','decoded_image','bytes')}
                    record = dict(doc_id=start + offset, doc=clean_doc, responses=responses)
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                handle.flush()
        dump(target.with_suffix('.complete.json'), dict(count=len(dataset), rollouts=rollouts,
             dataset_id=spec['repo'], dataset_revision=spec['revision'], dataset_fingerprint=dataset._fingerprint,
             sha256=hashlib.sha256(target.read_bytes()).hexdigest()))


def read_samples(args, task, repeat):
    path = sample_path(args, task, repeat)
    complete = json.loads(path.with_suffix('.complete.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != complete['sha256']:
        raise ValueError(f'Sample hash changed: {path}')
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    expected = min(args.limit, TASKS[task]['count']) if args.limit else TASKS[task]['count']
    rollouts = 3 if task == 'gaze' else 1
    if len(records) != expected or complete['count'] != expected or complete['rollouts'] != rollouts:
        raise ValueError(f'Incomplete {task} sample coverage')
    if [r['doc_id'] for r in records] != list(range(expected)) or any(len(r['responses']) != rollouts for r in records):
        raise ValueError(f'Invalid {task} document/rollout coverage')
    return records


def metric_summary(values):
    return dict(mean=statistics.mean(values), sem=statistics.stdev(values) / math.sqrt(len(values)) if len(values)>1 else None, repeats=values)


def score_gaze(args):
    import numpy as np
    from ggr_eval.gaze import calibration_ids, parse_predictions, score_grid, select_fit, SIGMAS, EPSILONS, UNIFORM_LPD, sem
    from ggr_eval.scorers.held_out_gaze import validate_fixations
    arrays, humans = [], None
    for repeat in range(args.repeats):
        records = read_samples(args, 'gaze', repeat)
        population = [validate_fixations(r['doc']['fixations'], mode='discard') for r in records]
        if humans is not None and population != humans:
            raise ValueError('Human fixation population differs across repeats')
        humans = population
        arrays.append(np.array([np.mean([score_grid(human, parse_predictions(raw, 'normalized_xy')) for raw in r['responses']], axis=0) for r,human in zip(records,population)]))
    n = len(humans)
    calibration = calibration_ids(n)
    evaluation = sorted(set(range(n)) - calibration)
    if len(calibration)<2 or len(evaluation)<2:
        return dict(status='partial: too few examples to fit gaze alignment')
    pooled = np.mean(arrays, axis=0)
    index = select_fit(pooled, calibration)
    sigma, epsilon = [(s,e) for s in SIGMAS for e in EPSILONS][index]
    bits = (pooled[evaluation,index] - UNIFORM_LPD) / math.log(2)
    return dict(mean=float(bits.mean()), sem=sem(bits), units='bits/fixation', sigma=sigma, epsilon=epsilon,
                calibration_examples=len(calibration), evaluation_examples=len(evaluation),
                calibration_doc_ids=sorted(calibration), evaluation_doc_ids=evaluation,
                human_fixations_sha256=hashlib.sha256(json.dumps(humans,separators=(',',':')).encode()).hexdigest())


def score(args, config, tasks):
    os.environ['JUDGE_MODEL_PATH'] = 'Qwen/Qwen3-32B'
    os.environ['CHARXIV_JUDGE_MODEL_PATH'] = 'Qwen/Qwen3-32B'
    os.environ['CHARXIV_JUDGE_TENSOR_PARALLEL_SIZE'] = str(args.judge_tensor_parallel_size)
    os.environ['CHARXIV_JUDGE_GPU_MEMORY_UTILIZATION'] = str(args.gpu_memory_utilization)
    os.environ['VLLM_JUDGE_MAX_NUM_SEQS'] = '32'
    os.environ['HELD_OUT_GAZE_COORD_FORMAT'] = 'normalized_xy'
    scores, counts = {}, {}
    for task in tasks:
        if task == 'gaze':
            continue
        spec, module = TASKS[task], module_for(task)
        values = []
        for repeat in range(args.repeats):
            records = read_samples(args, task, repeat)
            scored = [getattr(module,spec['score'])({**r['doc'], 'doc_id': str(r['doc_id'])}, r['responses']) for r in records]
            items = [item[spec['metric']] for item in scored]
            if spec.get('aggregate'):
                aggregate = getattr(module,spec['aggregate'])
                value = aggregate(items)
                details = getattr(aggregate, 'details', None)
                if details is not None:
                    if len(details) != len(records):
                        raise ValueError('Judge detail coverage does not match sample coverage')
                    dump(args.output/f'{task}.run{repeat+1}.judge.json', details)
            else:
                value = statistics.mean(items)
            dump(args.output/f'{task}.run{repeat+1}.scores.json', scored)
            counts[task] = len(records)
            values.append(100 * value)
        scores[task] = metric_summary(values)
    levels = ['visual_probe_easy','visual_probe_medium','visual_probe_hard']
    if all(t in scores for t in levels):
        scores['visual_probe'] = metric_summary([sum(scores[t]['repeats'][i] * counts[t] for t in levels)/sum(counts[t] for t in levels) for i in range(args.repeats)])
    six = ['chartqa_pro','chartqa','countbenchqa','countqa','mme_realworld_lite','visual_probe']
    if all(t in scores for t in six):
        scores['overall'] = metric_summary([statistics.mean(scores[t]['repeats'][i] for t in six) for i in range(args.repeats)])
    if 'gaze' in tasks:
        scores['gaze_alignment'] = score_gaze(args)
    result = dict(condition=args.condition, model=config['model_id'], partial=bool(args.limit), counts=counts, metrics=scores)
    dump(args.output/'results.json', result)
    print(json.dumps(result, indent=2))


def main():
    args = parser().parse_args()
    if args.repeats<1 or args.batch_size<1 or args.limit is not None and args.limit<1:
        raise ValueError('Repeats, batch size and limit must be positive')
    if args.repeat_index is not None and not 0 <= args.repeat_index < args.repeats:
        raise ValueError('Repeat index must be between zero and repeats minus one')
    config, prompt = configuration(args)
    tasks = selected_tasks(args)
    protocol = dict(condition=args.condition, **config, tasks={t:TASKS[t] for t in tasks}, repeats=args.repeats,
                    limit=args.limit, seed=args.seed, repeat_seeds=[args.seed + i for i in range(args.repeats)],
                    batch_size=args.batch_size, generation_settings=sampling_kwargs(RESPONSE_TOKENS),
                    judge=dict(model='Qwen/Qwen3-32B',temperature=.7,top_p=.8,top_k=20,min_p=0.,max_tokens=2048,max_model_len=18384,max_num_seqs=32),
                    gaze=dict(rollouts_per_trial=3,split_seed=314159,sigmas=[25,35,50,70,100,140,200],epsilons=[.01,.03,.1,.2,.4,.7,.9]))
    if args.dry_run:
        print(json.dumps(protocol, indent=2)); return
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = args.output/'protocol.json'
    if metadata.exists() and json.loads(metadata.read_text()) != protocol:
        raise ValueError('Output directory belongs to a different evaluation protocol')
    dump(metadata,protocol)
    if args.stage in ('all','generate') and args.repeat_index is None:
        # Generation processes exit before the judge loads, releasing actor VRAM.
        argv = sys.argv[1:]
        for repeat in range(args.repeats):
            subprocess.run([sys.executable,__file__,*argv,'--stage','generate','--repeat-index',str(repeat)],check=True)
        if args.stage=='all':
            subprocess.run([sys.executable,__file__,*argv,'--stage','score'],check=True)
    elif args.stage=='generate':
        generate(args,config,prompt,tasks)
    else:
        score(args,config,tasks)

if __name__=='__main__':
    main()
