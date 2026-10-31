"""Benchmark definitions for grounded reasoning evaluation."""
from importlib import import_module
from ggr_eval.generation import RESPONSE_TOKENS
import json
from pathlib import Path

TASKS = {
    'chartqa_pro': dict(repo='ahmed-masry/ChartQAPro', split='test', module='chartqa_pro', text='chartqa_pro_doc_to_text_alt', visual='chartqa_pro_doc_to_visual', score='chartqa_pro_process_results', metric='relaxed_overall', tokens=RESPONSE_TOKENS, count=1948),
    'chartqa': dict(repo='lmms-lab/ChartQA', split='test', module='chartqa', text='chartqa_doc_to_text', visual='chartqa_doc_to_visual', score='chartqa_process_results', metric='relaxed_overall', tokens=RESPONSE_TOKENS, count=2500),
    'countbenchqa': dict(repo='vikhyatk/CountBenchQA', split='test', module='countbenchqa', text='countbenchqa_doc_to_text', visual='countbenchqa_doc_to_visual', score='countbenchqa_cot_process_results', metric='exact_match', tokens=RESPONSE_TOKENS, count=491),
    'countqa': dict(repo='Jayant-Sravan/CountQA', split='test', module='countqa', process='countqa_process_docs', text='countqa_doc_to_text', visual='countqa_doc_to_visual', score='countqa_cot_process_results', metric='exact_match', tokens=RESPONSE_TOKENS, count=1528),
    'mme_realworld_lite': dict(repo='yifanzhang114/MME-RealWorld-lite-lmms-eval', split='train', module='mme_realworld', text='mme_realworld_doc_to_text_cot', visual='mme_realworld_doc_to_visual', score='mme_realworld_process_results', metric='mme_realworld_score', aggregate='mme_realworld_aggregate_results', tokens=RESPONSE_TOKENS, count=1919, post_prompt='\nSelect the best answer to the above multiple-choice question based on the image. After reasoning, end with \\boxed{A}, replacing A with the correct option letter (A, B, C, D, or E).'),
    **{f'visual_probe_{level.lower()}': dict(repo=f'Mini-o3/VisualProbe_{level}', split='validation', module='visual_probe', text='visual_probe_doc_to_text', visual='visual_probe_doc_to_visual', score='visual_probe_process_results', metric='judge_accuracy', aggregate='visual_probe_aggregate_judge_results', tokens=RESPONSE_TOKENS, count=count) for level, count in [('Easy',141), ('Medium',268), ('Hard',106)]},
    'gaze': dict(repo='gsarch/held_out_gaze', split='train', module='held_out_gaze', text='held_out_gaze_doc_to_text', visual='held_out_gaze_doc_to_visual', tokens=RESPONSE_TOKENS, count=698, pre_prompt='You are given an image and a visual question. Answer the question by first writing a grounded reasoning trace inside <think> and </think> tags. In the trace, include the image locations you inspect as normalized [x, y] coordinates in the 0 to 1000 coordinate system. After </think>, give the final answer, preferably in \\boxed{...}. Question: '),
}

for _task, _data in json.loads((Path(__file__).resolve().parent / 'datasets.json').read_text()).items():
    TASKS[_task]['revision'] = _data['revision']

def module_for(task):
    return import_module('ggr_eval.scorers.' + TASKS[task]['module'])

def load_task(task, limit=None):
    from datasets import load_dataset
    spec = TASKS[task]
    dataset = load_dataset(spec['repo'], split=spec['split'], revision=spec['revision'])
    if spec.get('process'):
        dataset = getattr(module_for(task), spec['process'])(dataset)
    if len(dataset) != spec['count']:
        raise ValueError(f"{task}: expected {spec['count']} examples, found {len(dataset)}; check the dataset revision")
    return dataset.select(range(min(limit, len(dataset)))) if limit else dataset
