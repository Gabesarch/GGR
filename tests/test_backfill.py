"""Offline checks for condition identity, filtering, and portable SFT data."""
from collections import Counter
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backfill.build_sft import format_candidate
from backfill.formatting import balance_unique_samples_by_category
from backfill.generate import ERROR_FLAGS, parse_judgment, passes, process_trial, teacher_prompt
from backfill.normalize_sft_point_text import normalize_text

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class BackfillTest(unittest.TestCase):
    def test_teacher_conditioning(self):
        row = {'question': 'What color?', 'answer': 'SECRET_REFERENCE',
               'fixation_list': [[100, 200], [100, 200], [-1, 1100]]}
        ggr = teacher_prompt(row, 'ggr')
        no_gaze = teacher_prompt(row, 'no_gaze')
        self.assertNotIn(row['answer'], ggr + no_gaze)
        self.assertEqual(ggr.count('- (100, 200)'), 2)
        self.assertIn('- (-1, 1100)', ggr)
        self.assertNotIn('human-attended', no_gaze)
        self.assertNotIn('(100, 200)', no_gaze)

    def test_ggr_judge_fails_closed(self):
        result = {'aligned': True, 'visually_correct': True, 'quality_score': 4, 'reason': 'ok',
                  **{flag: False for flag in ERROR_FLAGS}}
        self.assertTrue(passes(parse_judgment(json.dumps(result))))
        for flag in ERROR_FLAGS:
            wrong = {**result, flag: True}
            self.assertFalse(passes(wrong))
        with self.assertRaises(ValueError):
            parse_judgment(json.dumps({**result, 'aligned': 'true'}))
        with self.assertRaises(ValueError):
            parse_judgment('{"accept": 1}')

    def test_shared_condition_filter(self):
        result = {'aligned': True, 'visually_correct': True, 'quality_score': 4, 'reason': 'ok',
                  **{flag: False for flag in ERROR_FLAGS}}
        trial = {'sample_id': 's:1', 'question': 'Count?', 'ground_truth_answer': '2',
                 'category': 'counting', 'cohort': 'new_counting', 'image_pixel_sha256': 'abc'}
        response = r'Two objects at (20, 30) and (40, 50). \boxed{2}'
        outputs = []
        for condition in ['ggr', 'no_gaze']:
            sample = {'sample_idx': 0, 'status': 'complete', 'thinking_closed': None,
                      'accepted': passes(parse_judgment(json.dumps(result))), 'text': response}
            outputs.append(format_candidate({**trial, 'condition': condition}, sample, 'same system'))
        self.assertTrue(all(r['metadata']['accepted'] for r in outputs))
        self.assertEqual(outputs[0]['messages'], outputs[1]['messages'])
        for flag in ERROR_FLAGS:
            rejection = {**result, flag: True}
            decisions = [passes(parse_judgment(json.dumps(rejection))) for _ in ['ggr', 'no_gaze']]
            self.assertEqual(decisions, [False, False])

    def test_requests_share_sampling_and_judging(self):
        response = r'The target is at (20, 30). \boxed{red}'
        judgment = {'aligned': True, 'visually_correct': True, 'quality_score': 4,
                    'reason': 'ok', **{flag: False for flag in ERROR_FLAGS}}
        row = {'sample_id': 's:1', 'question': 'What color?', 'answer': 'red',
               'cohort': 'old_search', 'fixation_list': [[20, 30]],
               'image_pixel_sha256': 'abc', 'image': None}
        calls = {}
        with tempfile.TemporaryDirectory() as directory:
            for condition in ['ggr', 'no_gaze']:
                calls[condition] = []
                def create(**request):
                    calls[condition].append(request)
                    text = response if request['n'] == 2 else json.dumps(judgment)
                    return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
                        message=SimpleNamespace(content=text)) for _ in range(request['n'])])
                client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
                fake_openai = SimpleNamespace(OpenAI=lambda **kwargs: client)
                args = SimpleNamespace(output=Path(directory)/condition, condition=condition,
                    base_url='http://localhost:8000/v1', judge_base_url='http://localhost:8000/v1',
                    api_key_env='GGR_TEST_UNUSED_KEY', teacher_model='Qwen/test', judge_model='Qwen/test', samples=2)
                with patch.dict('sys.modules', {'openai': fake_openai}), patch('backfill.generate.image_data', return_value='image-data'):
                    self.assertEqual(process_trial(row, args), (2, 2))
                with patch.dict('sys.modules', {'openai': fake_openai}), patch('backfill.generate.image_data', return_value='image-data'):
                    self.assertEqual(process_trial(row, args), (2, 2))
                self.assertEqual(len(calls[condition]), 3, 'Completed trials must resume without requests')
        teacher_requests = [{k: v for k, v in calls[c][0].items() if k != 'messages'} for c in ['ggr', 'no_gaze']]
        self.assertEqual(teacher_requests[0], teacher_requests[1])
        self.assertEqual(calls['ggr'][1:], calls['no_gaze'][1:])
        self.assertEqual(calls['ggr'][0]['n'], 2)
        self.assertEqual(calls['ggr'][1]['max_tokens'], 8000)
        self.assertEqual(calls['ggr'][1]['temperature'], 0.6)

    def test_formatting_coordinates_and_original_answer(self):
        trial = {'sample_id': 's:1', 'question': 'Count?', 'ground_truth_answer': 'ORIGINAL',
                 'category': 'counting', 'cohort': 'new_counting', 'condition': 'ggr',
                 'image_pixel_sha256': 'abc'}
        sample = {'sample_idx': 0, 'status': 'complete', 'thinking_closed': None,
                  'accepted': True, 'text': r'The fixation list suggests [1.5, 2.5]. \boxed{teacher answer}'}
        row = format_candidate(trial, sample, 'Answer with (x, y) evidence.')
        self.assertEqual(row['gt_answer'], 'ORIGINAL')
        self.assertIn(r'\boxed{teacher answer}', row['messages'][2]['content'])
        self.assertNotIn('fixation', row['messages'][2]['content'].lower())
        self.assertIn('(2, 3)', row['messages'][2]['content'])
        self.assertEqual(normalize_text('[786, 444] to [816, 491]')[0], '(801, 468)')

    def test_balancing_caps_and_coverage(self):
        rows = []
        for category, groups in [('chart', 3), ('counting', 4), ('search', 2)]:
            for group in range(groups):
                for sample in range(30):
                    rows.append({'id': f'{category}:{group}:{sample}', 'metadata': {
                        'category': category, 'count_target': 7,
                        'question_image_key_sha256': f'{category}:{group}'}})
        selected, audit = balance_unique_samples_by_category(rows, seed=20260830, max_samples_per_question=25)
        counts = Counter(r['metadata']['category'] for r in selected)
        groups = Counter(r['metadata']['question_image_key_sha256'] for r in selected)
        self.assertLessEqual(max(counts.values()) / min(counts.values()), 1.25)
        self.assertLessEqual(max(groups.values()), 25)
        self.assertEqual(len(groups), 9)
        self.assertEqual(len({r['id'] for r in selected}), len(selected))

    def test_portable_packaging_and_registration(self):
        uploader = module('upload_sft_dataset')
        preparer = module('prepare_sft_data')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            (source / 'picture.png').write_bytes(b'fake image bytes for packaging')
            row = {'id': 'example', 'messages': [{'role': 'system', 'content': 'old prompt'},
                   {'role': 'user', 'content': '<image>\nquestion'},
                   {'role': 'assistant', 'content': 'reasoning'}], 'images': ['picture.png'],
                   'gt_answer': 'original', 'metadata': {'category': 'chart', 'source_path': str(source)}}
            raw = source / 'train.json'
            raw.write_text(json.dumps([row]))
            spec = {'source_file': 'train.json', 'source_sha256': uploader.sha256(raw), 'rows': 1}
            package = root / 'package'
            package.mkdir()
            result = uploader.package_condition('ggr', spec, source, source, package, {})
            actual = json.loads((package / result['file']).read_text())[0]
            self.assertEqual(actual['gt_answer'], row['gt_answer'])
            self.assertEqual(actual['messages'][1:], row['messages'][1:])
            self.assertEqual(actual['messages'][0]['content'], (ROOT / 'prompts/ggr.txt').read_text().strip())
            self.assertNotIn('source_path', actual['metadata'])
            self.assertFalse(Path(actual['images'][0]).is_absolute())
            registry = root / 'data/dataset_info.json'
            preparer.register(registry, 'ggr', package / result['file'])
            preparer.register(registry, 'ggr', package / result['file'])
            with self.assertRaises(ValueError):
                preparer.register(registry, 'ggr', root / 'different.json')
            with self.assertRaises(ValueError):
                preparer.safe_relative('../outside')


if __name__ == '__main__':
    unittest.main()
