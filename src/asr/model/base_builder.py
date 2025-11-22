# src/asr/model/base_builder.py

from typing import Optional, Tuple

import torch as T
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from asr.configs.main_conf import MODEL_NAME, LANG, TASK


class BaseModelBuilder:
    """
    Helper class to build the *base* (non-fine-tuned) Whisper model
    and its processor with a consistent generation config.
    """

    @staticmethod
    def build_base_model(
        device: Optional[str] = None,
        eval_mode: bool = True,
    ) -> Tuple[WhisperForConditionalGeneration, WhisperProcessor]:
        """
        Build the base Whisper model and processor.

        Args:
            device: Optional device string (e.g., "cuda", "cpu").
                    If provided, the model will be moved to this device.
            eval_mode: If True, puts the model in eval() mode.

        Returns:
            (model, processor)
        """
        # Load processor (tokenizer + feature extractor) from pretrained name
        processor = WhisperProcessor.from_pretrained(
            MODEL_NAME,
            language=LANG,
            task=TASK,
        )

        # Load base Whisper model (no fine-tuning weights)
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

        # Configure generation settings
        gc = model.generation_config
        gc.language = LANG
        gc.task = TASK
        gc.do_sample = False
        gc.num_beams = 1
        gc.no_repeat_ngram_size = 4
        gc.repetition_penalty = 1.1
        gc.max_new_tokens = 225

        # Move to device if requested
        if device is not None:
            model = model.to(device)

        # Use eval mode for inference (disable dropout, etc.)
        if eval_mode:
            model.eval()

        return model, processor
