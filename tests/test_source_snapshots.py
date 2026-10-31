"""Canonical SFT snapshots and explicit source paths preserve hash validation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/upload_sft_dataset.py'
SOURCES = {'ggr': 'ggr/train.json', 'no_gaze': 'no_gaze/train.json',
           'validation': 'validation/validation.json'}


class SnapshotSourceTests(unittest.TestCase):
    def fixture(self, root):
        snapshots = root / 'snapshots'
        images = root / 'images'
        images.mkdir()
        (images / 'image.png').write_bytes(b'image payload')
        conditions = {}
        paths = {}
        for name, relative in SOURCES.items():
            path = snapshots / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps([{
                'id': name, 'messages': [
                    {'role': 'system', 'content': 'system'},
                    {'role': 'user', 'content': '<image>\nQuestion'},
                    {'role': 'assistant', 'content': 'Answer'}],
                'images': ['image.png'], 'gt_answer': 'Answer',
                'metadata': {'category': 'chart'},
            }]))
            conditions[name] = {'source_file': relative, 'rows': 1,
                                'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            paths[name] = path
        manifest = root / 'manifest.json'
        manifest.write_text(json.dumps({'conditions': conditions}))
        return snapshots, images, manifest, paths

    def command(self, *arguments):
        return subprocess.run([sys.executable, str(SCRIPT), *map(str, arguments)],
                              capture_output=True, text=True)

    def test_manifest_canonical_paths(self):
        manifest = json.loads((ROOT / 'backfill/paper_data.json').read_text())
        self.assertEqual({name: spec['source_file'] for name, spec in manifest['conditions'].items()}, SOURCES)

    def test_cli_packages_canonical_and_explicit_sources(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                snapshots, images, manifest, paths = self.fixture(root)
                arguments = ['--manifest', manifest, '--image-root', images,
                             '--prepare-only', '--output', root / 'package']
                if explicit:
                    for name, path in paths.items():
                        renamed = root / f'{name} source=annotation.json'
                        path.rename(renamed)
                        arguments.extend(['--source', f'{name}={renamed}'])
                else:
                    arguments.extend(['--source-root', snapshots])
                result = self.command(*arguments)
                self.assertEqual(result.returncode, 0, result.stderr)
                packaged = json.loads((root / 'package/manifest.json').read_text())
                original = json.loads(manifest.read_text())['conditions']
                for name, relative in SOURCES.items():
                    rows = json.loads((root / 'package' / relative).read_text())
                    self.assertEqual(rows[0]['id'], name)
                    self.assertEqual(rows[0]['messages'][2]['content'], 'Answer')
                    self.assertEqual(packaged['configurations'][name]['source_sha256'],
                                     original[name]['source_sha256'])

    def test_checksum_rejection_for_both_source_modes(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                snapshots, _, manifest, paths = self.fixture(root)
                paths['ggr'].write_text('[]')
                arguments = ['--manifest', manifest, '--condition', 'ggr']
                if explicit:
                    arguments.extend(['--source', f'ggr={paths["ggr"]}'])
                else:
                    arguments.extend(['--source-root', snapshots])
                result = self.command(*arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('content hash changed', result.stderr)

    def test_invalid_source_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, manifest, paths = self.fixture(root)
            cases = [['ggr='], ['unknown=file'], ['no_gaze=file'],
                     [f'ggr={paths["ggr"]}', f'ggr={paths["ggr"]}']]
            for values in cases:
                with self.subTest(values=values):
                    arguments = ['--manifest', manifest, '--condition', 'ggr']
                    for value in values:
                        arguments.extend(['--source', value])
                    result = self.command(*arguments)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn('unique selected CONDITION=FILE', result.stderr)


if __name__ == '__main__':
    unittest.main()
