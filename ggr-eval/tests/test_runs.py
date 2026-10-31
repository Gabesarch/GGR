"""Actor repeat seeds and generation/scoring protocol integration without GPUs."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ggr_eval.tasks import TASKS

SCRIPT = Path(__file__).resolve().parents[1] / 'evaluate.py'
spec = importlib.util.spec_from_file_location('ggr_evaluation_runner', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class TrialDataset(list):
    _fingerprint = 'test-dataset'


class EvaluationRunTests(unittest.TestCase):
    def run_evaluation(self, output, seed=None, condition='ggr-sft'):
        actors, sampling, image_settings, commands = [], [], [], []
        dataset = TrialDataset([{'image': object(), 'number': 2, 'question': 'How many cats?'}])

        class Actor:
            def __init__(self, **kwargs):
                actors.append(kwargs)

            def chat(self, messages, sampling_params, use_tqdm):
                sampling.append(sampling_params)
                return [SimpleNamespace(outputs=[SimpleNamespace(text=r'\boxed{2}')]) for _ in messages]

        def messages(images, question, prompt, max_pixels, min_pixels):
            image_settings.append((min_pixels, max_pixels))
            return [{'role': 'user', 'content': question}]

        def child(command, check):
            self.assertTrue(check)
            commands.append(command)
            with patch.object(sys, 'argv', command[1:]):
                runner.main()

        argv = [str(SCRIPT), '--condition', condition, '--output', str(output),
                '--tasks', 'countbenchqa', '--limit', '1']
        if seed is not None:
            argv.extend(['--seed', str(seed)])
        fake_vllm = SimpleNamespace(LLM=Actor, SamplingParams=lambda **kwargs: kwargs)
        with patch.dict(sys.modules, {'vllm': fake_vllm}), \
             patch.object(runner, 'load_task', return_value=dataset), \
             patch('ggr_eval.generation.make_messages', side_effect=messages), \
             patch.object(runner.subprocess, 'run', side_effect=child), \
             redirect_stdout(io.StringIO()):
            with patch.object(sys, 'argv', argv):
                runner.main()
            protocol = json.loads((output / 'protocol.json').read_text())
            results = json.loads((output / 'results.json').read_text())
            with patch.object(sys, 'argv', argv + ['--stage', 'score']):
                runner.main()
            self.assertEqual(json.loads((output / 'protocol.json').read_text()), protocol)
            self.assertEqual(json.loads((output / 'results.json').read_text()), results)

        expected = [1, 2, 3] if seed is None else [seed + i for i in range(3)]
        self.assertEqual([actor['seed'] for actor in actors], expected)
        self.assertEqual(protocol['repeat_seeds'], expected)
        self.assertEqual(protocol['generation_settings']['max_tokens'], 14336)
        self.assertEqual([params['max_tokens'] for params in sampling], [14336] * 3)
        self.assertEqual(image_settings, [(2000000, 10000000)] * 3)
        self.assertEqual(results['metrics']['countbenchqa']['repeats'], [100.0] * 3)
        self.assertEqual(len(commands), 4)
        return [actor['seed'] for actor in actors]

    def test_repeat_seeds_and_protocol_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            defaults = self.run_evaluation(root / 'default')
            self.assertEqual(self.run_evaluation(root / 'repeated'), defaults)
            for condition in runner.CONDITIONS:
                with self.subTest(condition=condition):
                    self.run_evaluation(root / condition, seed=37, condition=condition)

    def test_all_tasks_use_response_budget(self):
        self.assertEqual({task['tokens'] for task in TASKS.values()}, {14336})

    def test_invalid_repeat_index(self):
        for index in (-1, 3):
            with self.subTest(index=index), patch.object(sys, 'argv', [
                str(SCRIPT), '--condition', 'ggr-sft', '--output', 'unused',
                '--repeat-index', str(index), '--dry-run',
            ]):
                with self.assertRaisesRegex(ValueError, 'Repeat index'):
                    runner.main()


if __name__ == '__main__':
    unittest.main()
