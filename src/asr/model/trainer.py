import pysrt
import torch as T
from datasets import Dataset
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          WhisperForConditionalGeneration, WhisperProcessor)

from asr.configs.main_conf import LANG, TASK, MODEL_NAME, MANIFEST, AUDIO_DIR, SUB_DIR, STR_SUFFIX


class WhisperSafeTrainer(Seq2SeqTrainer):
    """Custom Trainer: use base model under PEFT; feed input_features directly; stable generate."""

    def _peft_base(self, model):
        return getattr(model, "base_model", model)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        base = self._peft_base(model)
        input_features = inputs.pop("input_features")
        labels = inputs.get("labels")
        if self.args.gradient_checkpointing and isinstance(input_features, T.Tensor):
            input_features = input_features.requires_grad_(True)
        outputs = base(input_features=input_features, labels=labels)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        return (loss, outputs) if return_outputs else loss

    @T.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        base = self._peft_base(model)
        has_labels = "labels" in inputs
        input_features = inputs["input_features"]
        labels = inputs["labels"] if has_labels else None

        gen_kwargs = dict(
            language=LANG,
            task=TASK,
            do_sample=False,
            num_beams=1,
            no_repeat_ngram_size=4,
            repetition_penalty=1.1,
            max_new_tokens=self.args.generation_max_length or 225,
        )

        if self.args.predict_with_generate and not prediction_loss_only:
            generated_tokens = base.generate(
                input_features=input_features, **gen_kwargs
            )
        else:
            generated_tokens = None

        with self.compute_loss_context_manager():
            outputs = (
                base(input_features=input_features, labels=labels)
                if has_labels
                else base(input_features=input_features)
            )
            loss = (
                (
                    outputs["loss"] if isinstance(outputs, dict) else outputs.loss
                ).detach()
                if has_labels
                else None
            )

        if generated_tokens is not None and generated_tokens.ndim == 1:
            generated_tokens = generated_tokens.unsqueeze(0)

        return (loss, generated_tokens, labels if has_labels else None)

