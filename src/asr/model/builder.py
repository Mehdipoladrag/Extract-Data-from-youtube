from peft import LoraConfig, get_peft_model
from transformers import (
    WhisperForConditionalGeneration, 
    WhisperProcessor
)
from asr.configs.main_conf import LANG, TASK, MODEL_NAME, MANIFEST, AUDIO_DIR, SUB_DIR, STR_SUFFIX




class ModelBuilderProcessor:
    """Create Whisper processor/model, set language/task, disable cache (GC), apply LoRA."""

    @staticmethod
    def build_model():
        processor = WhisperProcessor.from_pretrained(MODEL_NAME)
        if hasattr(processor, "feature_extractor"):
            try:
                processor.feature_extractor.return_attention_mask = False
            except Exception:
                pass
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

        model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
        model.generation_config.language = LANG
        model.generation_config.task = TASK
        model.config.use_cache = False

        peft_cfg = LoraConfig(
            task_type="SEQ_2_SEQ_LM",
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()
        return model, processor
