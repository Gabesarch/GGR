"""Check worker setup with the composed release configurations, without GPUs."""
import os
from pathlib import Path
import unittest

from hydra import compose, initialize_config_dir

from verl.trainer.main_ppo import TaskRunner


ROOT = Path(__file__).resolve().parents[2]


class CriticSetupTests(unittest.TestCase):
    def config(self, condition, overrides=None):
        os.environ["GGR_ROOT"] = str(ROOT)
        with initialize_config_dir(config_dir=str(ROOT / "ggr-rl/configs"), version_base=None):
            return compose(config_name=condition, overrides=overrides or [])

    def test_release_conditions_start_without_a_critic(self):
        for condition in ("ggr", "no_gaze"):
            with self.subTest(condition=condition):
                runner = TaskRunner()
                runner.add_critic_worker(self.config(condition))
                self.assertEqual(runner.role_worker_mapping, {})
                self.assertEqual(runner.mapping, {})

    def test_critic_requests_are_rejected(self):
        for overrides in (["critic.enable=true"], ["algorithm.adv_estimator=gae"]):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, "has no critic"):
                    TaskRunner().add_critic_worker(self.config("ggr", overrides))


if __name__ == "__main__":
    unittest.main()
