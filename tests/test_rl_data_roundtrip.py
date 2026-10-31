"""Portable dataset packaging preserves the values consumed by the RL scorer."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


upload = load_script('upload_rl_dataset')
prepare = load_script('prepare_rl_data')


class DatasetRoundtrip(unittest.TestCase):
    def test_embedded_image_and_reward_fields(self):
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / 'image.png'
            image.write_bytes(b'original-image-bytes')
            row = {
                'data_source': 'qwen3vl_filtering', 'ability': 'chart',
                'prompt': [{'role': 'user', 'content': '/Do not rewrite this question'}],
                'images': [str(image)],
                'reward_model': {'style': 'rule', 'ground_truth': '/literal answer'},
                'extra_info': {'reward_type': 'string_match', 'answer': '/literal answer',
                               'metadata': {'source_jsonl': str(root / 'original.jsonl'), 'boxes': [[1, 2, 3, 4]]}},
                'avg_pass_rate': 0.0,
            }
            source = root / 'train.jsonl'
            source.write_text(json.dumps(row) + '\n')
            package = root / 'package'
            upload.write_split(source, 'train', package, None)
            example = pq.read_table(package / 'data/train-00000.parquet').to_pylist()[0]
            restored = prepare.restore_row(example, root / 'restored')
            for key in ('prompt', 'reward_model', 'data_source', 'ability', 'avg_pass_rate'):
                self.assertEqual(restored[key], row[key])
            self.assertEqual(restored['extra_info']['answer'], '/literal answer')
            self.assertEqual(restored['extra_info']['reward_type'], 'string_match')
            self.assertEqual(restored['extra_info']['metadata']['boxes'], [[1, 2, 3, 4]])
            self.assertEqual(restored['extra_info']['metadata']['source_jsonl'], 'original.jsonl')
            self.assertEqual((root / 'restored' / restored['images'][0]).read_bytes(), image.read_bytes())

    def test_bad_image_checksum_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, 'checksum'):
                prepare.restore_row({'record': json.dumps({'images': ['images/incorrect.png']}),
                                     'images': [{'bytes': b'wrong'}]}, Path(temporary))

    def test_unsafe_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            for reference in ('../escape.png', ''):
                with self.assertRaisesRegex(ValueError, 'unsafe'):
                    prepare.restore_row({'record': json.dumps({'images': [reference]}),
                                         'images': [{'bytes': b'wrong'}]}, Path(temporary))


if __name__ == '__main__':
    unittest.main()
