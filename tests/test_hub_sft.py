import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_sft', ROOT / 'scripts/prepare_sft_data.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class HubSFTTests(unittest.TestCase):
    def test_pinned_image_download_and_row_preservation(self):
        payload = b'image payload for checksum verification'
        name = hashlib.sha256(payload).hexdigest() + '.png'
        example = {'id': 'example', 'messages': [
            {'role': 'system', 'content': 'original'},
            {'role': 'user', 'content': '<image>\nQuestion'},
            {'role': 'assistant', 'content': 'Answer'}],
            'images': [{'path': f'https://huggingface.co/datasets/gsarch/Gaze-SFT/resolve/main/images/{name}', 'bytes': None}],
            'gt_answer': '/original answer', 'metadata_json': '{"sample_idx": 2}',
            'extra_fields_json': '{"weight": 0.5}'}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / 'source'
            source.write_bytes(payload)
            calls = []
            def fetch(*args, **kwargs):
                calls.append((args, kwargs))
                return source
            row = prepare.restore_parquet_row(example, 'gsarch/Gaze-SFT', 'pinned', directory / 'images', fetch)
            self.assertEqual(calls[0][1]['revision'], 'pinned')
            self.assertEqual(row['gt_answer'], example['gt_answer'])
            self.assertEqual(row['messages'][1:], example['messages'][1:])
            self.assertEqual(row['metadata'], {'sample_idx': 2})
            self.assertEqual(row['weight'], 0.5)
            self.assertEqual(row['messages'][0]['content'], (ROOT / 'prompts/ggr.txt').read_text().strip())
            self.assertEqual((ROOT / row['images'][0]).read_bytes(), payload)

    def test_foreign_or_traversing_image_urls_rejected(self):
        for value in ['https://example.com/image.png', '../image.png',
                      'https://huggingface.co/datasets/other/data/resolve/main/images/image.png',
                      'https://huggingface.co/datasets/gsarch/Gaze-SFT/resolve/main/images/%2e%2e/%2e%2e/file']:
            with self.assertRaises(ValueError):
                prepare.image_reference(value, 'gsarch/Gaze-SFT')


if __name__ == '__main__':
    unittest.main()
