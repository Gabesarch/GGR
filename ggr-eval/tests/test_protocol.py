import base64
import io
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from ggr_eval.gaze import calibration_ids, parse_predictions, score_grid, select_fit, UNIFORM_LPD
from ggr_eval.generation import RESPONSE_TOKENS, make_messages, actor_kwargs, sampling_kwargs
from ggr_eval.scorers.answer_extraction import extract_final_answer
from ggr_eval.scorers import chartqa, chartqa_pro, countbenchqa, countqa, mme_realworld, visual_probe


class ProtocolTests(unittest.TestCase):
    def test_saved_answer_forms(self):
        examples = [(r'<think>look at (100, 200)</think>\boxed{12}', '12'),
                    (r'\boxed{old} and \boxed{new}', 'new'),
                    ('<think>wrong answer: 99</think><answer>seven</answer>', 'seven')]
        for text, expected in examples:
            self.assertEqual(extract_final_answer(text), expected)

    def test_numeric_and_count_scorers(self):
        self.assertEqual(chartqa.chartqa_process_results({'type':'human_test','answer':['100']}, [r'\boxed{104}'])['relaxed_overall'],1)
        self.assertEqual(chartqa.chartqa_process_results({'type':'human_test','answer':['100']}, [r'\boxed{106}'])['relaxed_overall'],0)
        self.assertEqual(countbenchqa.countbenchqa_cot_process_results({'number':7,'question':'How many cats?'}, [r'<think>2 at (7, 8)</think>\boxed{seven}'])['exact_match'],1)
        self.assertEqual(countqa.countqa_cot_process_results({'answer':'18','question':'How many tiles?'}, [r'\boxed{17}'])['exact_match'],0)
        self.assertEqual(chartqa_pro.chartqa_pro_process_results({'Answer':['42'],'Question Type':'Factoid'},[r'\boxed{42}'])['relaxed_overall'],1)

    def test_visual_probe_parser(self):
        for output, expected in [('yes',True), ('{"is_equivalent":false}',False), ('invalid',False)]:
            self.assertEqual(visual_probe._parse_judge_output(output),expected)

    def test_judge_effective_sampling(self):
        # The paper's Qwen3-specific backend sets .7, independent of config's nominal temperature.
        import sys
        from types import SimpleNamespace
        from ggr_eval.scorers.vllm_judge import JudgeConfig, VLLMJudgeEngineQwen3
        fake_llm = SimpleNamespace(get_tokenizer=lambda: SimpleNamespace(chat_template=None))
        fake_vllm = SimpleNamespace(LLM=lambda **kwargs: fake_llm, SamplingParams=lambda **kwargs: kwargs)
        config = JudgeConfig('Qwen/Qwen3-32B',1,'bfloat16',18384,32,0.,.8,20,0.,2048,.9,'')
        with patch.dict(sys.modules, {'vllm':fake_vllm}):
            engine=VLLMJudgeEngineQwen3(config)
            self.assertEqual(engine._base_sampling_kwargs,dict(temperature=.7,top_p=.8,max_tokens=2048,top_k=20,min_p=0.))
            self.assertIn('im_start',engine._chat_template)

    def test_image_messages(self):
        from qwen_vl_utils import fetch_image
        image=Image.new('RGB',(100,50),'blue')
        messages=make_messages([image],'question','system',10000000,2000000)
        encoded=messages[1]['content'][0]['image_url']['url'].split(',',1)[1]
        actual=Image.open(io.BytesIO(base64.b64decode(encoded)))
        expected=fetch_image({'image':image,'max_pixels':10000000,'min_pixels':2000000})
        self.assertEqual(actual.size,expected.size)
        self.assertEqual(actual.tobytes(),expected.tobytes())
        self.assertEqual(messages[1]['content'][-1],{'type':'text','text':'question'})
        self.assertEqual(messages[0],{'role':'system','content':[{'type':'text','text':'system'}]})
        with self.assertRaises(ValueError):make_messages([],'question','system',10000000,2000000)

    def test_actor_protocol(self):
        self.assertEqual(sampling_kwargs(RESPONSE_TOKENS),dict(max_tokens=14336,temperature=1.,top_p=.95,top_k=20,presence_penalty=1.5))
        config=actor_kwargs('example/model',1,.9)
        self.assertEqual(config['seed'],1)
        self.assertEqual(config['max_num_batched_tokens'],8196)
        self.assertFalse(config['enable_prefix_caching'])

    def test_saved_score_entrypoint_and_integrity(self):
        import hashlib
        import json
        from pathlib import Path
        import subprocess
        import sys
        import tempfile
        script = Path(__file__).resolve().parents[1] / 'evaluate.py'
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            sample = output / 'chartqa.run1.jsonl'
            rows = [dict(doc_id=i, doc={'type':'human_test','answer':['100']}, responses=[r'\boxed{'+answer+'}']) for i,answer in enumerate(['104','106'])]
            sample.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            sample.with_suffix('.complete.json').write_text(json.dumps(dict(count=2,rollouts=1,sha256=hashlib.sha256(sample.read_bytes()).hexdigest())))
            command = [sys.executable,str(script),'--condition','ggr-sft','--output',directory,'--tasks','chartqa','--repeats','1','--limit','2','--stage','score']
            completed = subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stderr)
            result = json.loads((output/'results.json').read_text())
            self.assertEqual(result['metrics']['chartqa']['mean'],50.)
            self.assertTrue(result['partial'])
            self.assertNotIn('overall',result['metrics'])
            sample.write_text(sample.read_text()+'\n')
            completed = subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(completed.returncode,0)
            self.assertIn('Sample hash changed',completed.stderr)

    def test_gaze_population_and_density(self):
        calibration=calibration_ids(698)
        self.assertEqual(len(calibration),336)
        self.assertEqual(len(set(range(698))-calibration),362)
        self.assertEqual(parse_predictions('<think>(20,30) [20,30] (1001,0)</think> (500,500)','normalized_xy'),[[20.,30.],[20.,30.]])
        np.testing.assert_array_equal(score_grid([[20,30]],[]),np.full(49,UNIFORM_LPD))
        self.assertTrue(np.all(score_grid([[20,30]],[[20,30]]) > UNIFORM_LPD))
        # Calibration selection must be independent of scoring-set values.
        scores=np.zeros((698,49)); scores[list(calibration),3]=1
        selected=select_fit(scores,calibration)
        scores[list(set(range(698))-calibration),42]=1e6
        self.assertEqual(select_fit(scores,calibration),selected)

if __name__=='__main__': unittest.main()
