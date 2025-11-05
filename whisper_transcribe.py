# -*- coding: utf-8 -*-
from pathlib import Path

import soundfile as sf
import torch as T
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class WhisperTranscriber:
    """Single-file Whisper inference with clean setup.

    Responsibilities:
    - Load processor/model once and pin to device.
    - Prepare 16kHz mono audio and build input_features.
    - Generate transcription with stable, repetition-resistant settings.

    Notes:
    - Attention mask is not required for Whisper labels; we explicitly disable it
      on the feature extractor to avoid spurious warnings.
    - We pass `language` and `task` via generation_config and generate().
    """

    def __init__(self, model_name: str = "openai/whisper-small", language: str = "fa"):
        """Initialize processor/model, device, and generation config."""
        self.language = language
        self.device = "cuda" if T.cuda.is_available() else "cpu"

        try:
            if T.cuda.is_available():
                T.backends.cuda.matmul.fp32_precision = "tf32"
                T.backends.cudnn.conv.fp32_precision = "tf32"
        except Exception:
            pass

        self.processor = WhisperProcessor.from_pretrained(model_name)
        if hasattr(self.processor, "feature_extractor"):
            try:
                self.processor.feature_extractor.return_attention_mask = False
            except Exception:
                pass

        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )
        self.model.eval()

        self.model.generation_config.language = self.language
        self.model.generation_config.task = "transcribe"

        gc = self.model.generation_config
        gc.do_sample = False
        gc.num_beams = 1
        gc.no_repeat_ngram_size = 4
        gc.repetition_penalty = 1.1
        gc.max_new_tokens = 225

    def _load_audio(self, wav_path: Path):
        """Read audio, enforce mono 16kHz float32; raise if constraints fail."""
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != 16000:
            raise ValueError(f"Audio must be 16kHz; got {sr}")
        return audio, sr

    def _features_from_audio(self, audio, sr):
        """Build input_features; use key access for version compatibility."""
        feat = self.processor.feature_extractor(
            audio, sampling_rate=sr, return_tensors="pt"
        )
        input_features = feat["input_features"].to(self.device)
        return input_features

    def transcribe(self, wav_path: Path) -> str:
        """End-to-end transcription for a single file path."""
        audio, sr = self._load_audio(wav_path)
        input_features = self._features_from_audio(audio, sr)

        with T.no_grad():
            pred_ids = self.model.generate(
                input_features,
                language=self.language,
                task="transcribe",
                do_sample=False,
                num_beams=self.model.generation_config.num_beams,
                no_repeat_ngram_size=self.model.generation_config.no_repeat_ngram_size,
                repetition_penalty=self.model.generation_config.repetition_penalty,
                max_new_tokens=self.model.generation_config.max_new_tokens,
            )

        text_list = self.processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        return text_list[0] if text_list else ""


if __name__ == "__main__":
    MODEL = "openai/whisper-small"
    LANG = "fa"
    AUDIO = Path(
        "audio/processed/چطور ساعت مطالعه‌ت رو از ۲  به ۱۰ ساعت در روز برسونی؟ (بدون خستگی و کاملا واقعی)_16k.wav"
    )

    app = WhisperTranscriber(model_name=MODEL, language=LANG)
    text = app.transcribe(AUDIO)
    print("Transcription==>>", text)
