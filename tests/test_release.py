"""Portable release contracts that can be checked without models or GPUs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("upload_models", ROOT / "scripts/upload_models.py")
upload_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upload_models)


class ReleaseTests(unittest.TestCase):
    def test_condition_scope_and_prompt_hashes(self):
        release = json.loads((ROOT / "configs/release.json").read_text())
        self.assertEqual(set(release["conditions"]), {
            "no-gaze-sft", "ggr-sft", "no-gaze-rl", "ggr-rl",
        })
        for entry in release["conditions"].values():
            prompt = ROOT / entry["prompt_file"]
            self.assertFalse(Path(entry["prompt_file"]).is_absolute())
            self.assertEqual(hashlib.sha256(prompt.read_bytes()).hexdigest(), entry["system_prompt_sha256"])
            self.assertTrue(entry["model_id"].startswith("gsarch/"))
            self.assertEqual(entry["prompt_file"], "prompts/ggr.txt")

    def test_default_template_preserves_explicit_override(self):
        import jinja2
        native = "{% for m in messages %}{{m.role}}:{{m.content}}|{% endfor %}"
        template = jinja2.Environment().from_string(upload_models.default_template(native, "gaze prompt"))
        self.assertEqual(template.render(messages=[{"role": "user", "content": "Q"}]),
                         "system:gaze prompt|user:Q|")
        self.assertEqual(template.render(messages=[{"role": "system", "content": "override"}, {"role": "user", "content": "Q"}]),
                         "system:override|user:Q|")

    def test_sft_registry_starts_empty(self):
        self.assertEqual(json.loads((ROOT / "ggr-sft/data/dataset_info.json").read_text()), {})

    def test_training_and_evaluation_limits(self):
        import yaml
        release = json.loads((ROOT / 'configs/release.json').read_text())
        for entry in release['conditions'].values():
            self.assertEqual(entry['min_resize_pixels'], 2000000)
            self.assertEqual(entry['max_image_pixels'], 10000000)
        for condition in ('ggr', 'no_gaze'):
            sft = yaml.safe_load((ROOT / f'ggr-sft/configs/{condition}.yaml').read_text())
            rl = yaml.safe_load((ROOT / f'ggr-rl/configs/{condition}.yaml').read_text())
            self.assertEqual(sft['image_min_pixels'], 2000000)
            self.assertEqual(sft['image_max_pixels'], 8388608)
            self.assertEqual(sft['cutoff_len'], 32768)
            self.assertEqual(rl['data']['min_pixels'], 2000000)
            self.assertEqual(rl['data']['max_pixels'], 10000000)
            self.assertEqual(rl['data']['max_response_length'], 14336)


if __name__ == "__main__":
    unittest.main()
