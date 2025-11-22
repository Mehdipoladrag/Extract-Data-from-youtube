from typing import Literal, Optional
from pathlib import Path

from peft import LoraConfig, get_peft_model, PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from asr.configs.main_conf import LANG, TASK, MODEL_NAME

LORA_ROOT_DIR = Path("runs/whisper-fa-lora/checkpoint-700")


def _resolve_lora_dir(root: Path) -> Path:

    if (root / "adapter_config.json").exists():
        print(f"[INFO] Found adapter in root: {root}")
        return root

    candidates = []
    for p in root.glob("checkpoint-*"):
        if (p / "adapter_config.json").exists():
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(f"No LoRA adapter found under {root}")

    best = sorted(candidates)[-1]
    print(f"[INFO] Using LoRA checkpoint: {best}")
    return best


class ModelBuilderProcessor:
    """Create Whisper processor/model, set language/task, apply LoRA (train or inference)."""

    @staticmethod
    def build_model(
        mode: Literal["train", "inference"] = "train",
        lora_dir: Optional[str] = None,
    ):
        if lora_dir is None:
            lora_dir = str(LORA_ROOT_DIR)

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

        if mode == "train":
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

        elif mode == "inference":
            try:
                resolved = _resolve_lora_dir(Path(lora_dir))
                model = PeftModel.from_pretrained(model, str(resolved))
                print(f"[INFO] Loaded LoRA adapter from: {resolved}")
            except Exception as e:
                print(f"[WARN] Could not load LoRA from {lora_dir}: {e}")
                print("[WARN] Using base model without fine-tuning.")

        else:
            raise ValueError(f"Unknown mode: {mode}")

        return model, processor
