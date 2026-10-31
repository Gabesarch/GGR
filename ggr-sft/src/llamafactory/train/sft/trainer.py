# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import json
import os
import shutil
from functools import partial
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import safetensors.torch
import torch
from transformers import PreTrainedModel, Seq2SeqTrainer
from transformers.trainer import TRAINING_ARGS_NAME, TRAINER_STATE_NAME
from transformers.utils import is_peft_available
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...model.model_utils.misc import get_save_pretrained_kwargs
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, sft_loss_func


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


logger = logging.get_logger(__name__)


def _is_truthy_env(value: Optional[str]) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _checkpoint_step(path: str) -> Optional[int]:
    name = os.path.basename(path.rstrip(os.sep))
    prefix = "checkpoint-"
    if not name.startswith(prefix):
        return None

    step = name[len(prefix) :]
    return int(step) if step.isdigit() else None


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        ref_model: Optional["torch.nn.Module"] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        # Configure FP8 environment if enabled
        training_args: TrainingArguments = kwargs.get("args")
        if training_args.fp8:
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()

        super().__init__(**kwargs)
        trainer_model = getattr(self, "model", kwargs.get("model"))
        self.use_liger_fused_linear_cross_entropy = bool(
            model_args is not None
            and model_args.enable_liger_kernel
            and getattr(getattr(trainer_model, "config", None), "model_type", None) == "qwen3_vl"
        )
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        self.ref_model = ref_model

        if ref_model is not None:
            from trl.models.utils import prepare_deepspeed, prepare_fsdp

            if getattr(self.accelerator.state, "deepspeed_plugin", None) is not None:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif getattr(self.accelerator.state, "fsdp_plugin", None) is not None:
                if self.accelerator.is_fsdp2:
                    from accelerate.utils.fsdp_utils import fsdp2_prepare_model

                    self.ref_model = fsdp2_prepare_model(self.accelerator, self.ref_model)
                else:
                    self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        elif finetuning_args.use_eaft_loss:
            from ..trainer_utils import eaft_loss_func

            self.compute_loss_func = lambda outputs, labels, num_items_in_batch=None: eaft_loss_func(
                outputs, labels, num_items_in_batch, finetuning_args.eaft_alpha
            )
        elif finetuning_args.use_asft_loss:
            from ..trainer_utils import asft_loss_func

            self.compute_loss_func = partial(
                asft_loss_func,
                asft_alpha=finetuning_args.asft_alpha,
            )
        elif processor is not None and not self.use_liger_fused_linear_cross_entropy:
            self.compute_loss_func = sft_loss_func

        if training_args.fp8 and hasattr(self, "accelerator"):  # verify FP8 status after trainer initialization
            verify_fp8_status(self.accelerator, training_args)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def _save_full_state_only_at_last_checkpoint(self) -> bool:
        return _is_truthy_env(os.getenv("LLAMAFACTORY_SAVE_FULL_STATE_ONLY_LAST_CHECKPOINT"))

    def _strip_training_state_from_checkpoint(self, checkpoint_dir: str) -> None:
        if not self.is_world_process_zero():
            return

        file_patterns = [
            "optimizer.pt",
            "optimizer.bin",
            "optimizer.pt_*",
            "rank*-of-*-optimizer.pt",
            "scheduler.pt",
            "scaler.pt",
            "rng_state*.pth",
            TRAINER_STATE_NAME,
            "latest",
        ]
        dir_patterns = [
            "global_step*",
        ]

        for pattern in file_patterns:
            for path in glob.glob(os.path.join(checkpoint_dir, pattern)):
                if os.path.isfile(path) or os.path.islink(path):
                    os.remove(path)

        for pattern in dir_patterns:
            for path in glob.glob(os.path.join(checkpoint_dir, pattern)):
                if os.path.isdir(path):
                    shutil.rmtree(path)

    def _strip_training_state_from_older_checkpoints(self, current_checkpoint_dir: str) -> None:
        if not self.is_world_process_zero():
            return

        current_step = _checkpoint_step(current_checkpoint_dir)
        if current_step is None:
            return

        run_dir = os.path.dirname(current_checkpoint_dir)
        for checkpoint_dir in glob.glob(os.path.join(run_dir, "checkpoint-*")):
            checkpoint_step = _checkpoint_step(checkpoint_dir)
            if checkpoint_step is not None and checkpoint_step < current_step:
                self._strip_training_state_from_checkpoint(checkpoint_dir)

    @override
    def _save_checkpoint(self, model, trial):
        if not self._save_full_state_only_at_last_checkpoint():
            return super()._save_checkpoint(model, trial)

        original_save_only_model = self.args.save_only_model
        self.args.save_only_model = False
        try:
            result = super()._save_checkpoint(model, trial)
        finally:
            self.args.save_only_model = original_save_only_model

        run_dir = self._get_output_dir(trial=trial)
        checkpoint_dir = os.path.join(run_dir, f"checkpoint-{self.state.global_step}")
        self._strip_training_state_from_older_checkpoints(checkpoint_dir)
        return result

    def save_final_full_state_checkpoint(self) -> None:
        original_save_only_model = self.args.save_only_model
        self.args.save_only_model = False
        try:
            super()._save_checkpoint(self.model, trial=None)
        finally:
            self.args.save_only_model = original_save_only_model

        checkpoint_dir = os.path.join(self._get_output_dir(trial=None), f"checkpoint-{self.state.global_step}")
        self._strip_training_state_from_older_checkpoints(checkpoint_dir)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.finetuning_args.use_asft_loss:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                )
                ref_logits = ref_outputs.logits
            outputs = model(**inputs)
            return self.compute_loss_func(outputs, inputs["labels"], ref_logits)
        elif self.use_liger_fused_linear_cross_entropy and model.training:
            # Transformers normally pops labels before invoking a custom SFT
            # loss. Keep them in the model call so Liger can fuse lm_head and
            # cross entropy without materializing full-sequence fp32 logits.
            model_inputs = dict(inputs)
            if num_items_in_batch is not None:
                model_inputs["num_items_in_batch"] = num_items_in_batch

            outputs = model(**model_inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    @override
    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[dict[str, Any]] = None) -> None:
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info_rank0(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,)
        if is_peft_available():
            from peft import PeftModel

            supported_classes = (PreTrainedModel, PeftModel)

        model_to_save = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        save_kwargs = get_save_pretrained_kwargs(model_to_save) if isinstance(model_to_save, PreTrainedModel) else {}

        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(model_to_save, supported_classes):
                model_to_save.save_pretrained(output_dir, state_dict=state_dict, **save_kwargs)
            else:
                logger.info_rank0("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                safetensors.torch.save_file(
                    state_dict,
                    os.path.join(output_dir, "model.safetensors"),
                    metadata={"format": "pt"},
                )
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict, **save_kwargs)

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        elif (
            self.data_collator is not None
            and hasattr(self.data_collator, "tokenizer")
            and self.data_collator.tokenizer is not None
        ):
            logger.info_rank0("Saving Trainer.data_collator.tokenizer by default as Trainer.processing_class is `None`")
            self.data_collator.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        input_ids_column = dataset["input_ids"]
        try:
            input_ids_list = input_ids_column.to_pylist()
        except AttributeError:
            input_ids_list = list(input_ids_column)

        decoded_inputs = self.processing_class.batch_decode(input_ids_list, skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
