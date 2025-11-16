import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
import evaluate
import numpy as np
import pysrt
import soundfile as sf
import torch as T
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          WhisperForConditionalGeneration, WhisperProcessor)

from asr.configs.main_conf import LANG, TASK, MODEL_NAME, MANIFEST, AUDIO_DIR, SUB_DIR, STR_SUFFIX
from asr.data.manifest_builder import ManifestBuilder
from asr.data.audio_loader import LoadVoice
from asr.model.builder import ModelBuilderProcessor
from asr.model.collator import DataCollatorSpeechSeq2Seq
from asr.model.metrics import ComputeMetrics
from asr.pipeline.training import WhisperTrainingPipeline


class MapExampler:
    """Convert one manifest record to model inputs: speech features + tokenized labels."""

    def __init__(self, processor: WhisperProcessor):
        self.processor = processor

    def mapping(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        a, sr = LoadVoice(rec).load_audio()
        x = self.processor.feature_extractor(a, sampling_rate=sr, return_tensors=None)
        feats = x["input_features"] if isinstance(x, dict) else x.input_features
        if isinstance(feats, np.ndarray) and feats.ndim == 3:
            feats = feats[0]
        feats = feats.tolist()
        y = self.processor.tokenizer(rec["text"], add_special_tokens=True)
        return {"input_features": feats, "labels": y["input_ids"]}


class WhisperFineTuningApp:
    """Thin app wrapper to assemble and run the training pipeline."""

    def __init__(self):
        self.manifest_path = MANIFEST
        self.output_dir = "runs/whisper-fa-lora"

    def build_manifest(self):
        mb = ManifestBuilder()
        mb.build_if_needed()

    def build_model_and_processor(self):
        return ModelBuilderProcessor.build_model()

    def map_example(self, rec, processor):
        return MapExampler(processor).mapping(rec)

    def run(self):
        pipeline = WhisperTrainingPipeline(
            manifest_path=self.manifest_path,
            build_manifest_fn=self.build_manifest,
            build_model_and_processor=self.build_model_and_processor,
            map_example_fn=self.map_example,
            data_collator_cls=DataCollatorSpeechSeq2Seq,
            compute_metrics_fn_factory=ComputeMetrics,
            output_dir=self.output_dir,
        )
        pipeline.run()

    def transcribe_segment(self, audio_path: str, start_sec: float = 0.0, duration_sec: float = 60.0):
        model, processor = ModelBuilderProcessor.build_model()
        model.eval()
        device = "cuda" if T.cuda.is_available() else "cpu"
        model.to(device)

        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != 16000:
            raise ValueError(f"Expected 16kHz audio, got {sr} for {audio_path}")

        n = len(audio)
        s = max(0, int(round(start_sec * sr)))
        e = min(n, s + int(round(duration_sec * sr)))
        if s >= n:
            print("[warn] start beyond audio length"); return ""
        audio = audio[s:e]

        feats = processor.feature_extractor(audio, sampling_rate=sr, return_tensors="pt")["input_features"].to(device)
        with T.inference_mode():
            ids = model.generate(
                input_features=feats,
                language=LANG, task=TASK,
                do_sample=False, num_beams=1, max_new_tokens=225
            )
        text = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
        print(f"\n--- TRANSCRIPT ({int(duration_sec)}s @ {int(start_sec)}s) ---\n{text}\n")
        return text


if __name__ == "__main__":
    INFER_AUDIO = ""       
    START_SEC   = 0.0      
    DURATION_S  = 60.0      

    app = WhisperFineTuningApp()
    if INFER_AUDIO:
        app.transcribe_segment(INFER_AUDIO, START_SEC, DURATION_S)
    else:
        app.run()