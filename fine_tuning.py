import os, json, random, re
import torch
import pysrt
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Union
from transformers import (
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    WhisperProcessor,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass
from datasets import Dataset, load_dataset, Audio   
from jiwer import cer


"""
    Main Configurations
"""

LANG = "fa"
MODEL_NAME = "openai/whisper-small"
SEED = 42
MANIFEST = Path("manifest.json")
AUDIO_DIR = Path("audi/processed")
SUB_DIR = Path("subtitle")
STR_SUFFIX = [".fa.srt", ".srt"]

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


class SRTPreprocessor:
    @staticmethod
    def srt_time_to_sec(t) -> float:
        return t.hours * 3600 + t.minutes * 60 + t.seconds + t.milliseconds / 1000.0

    @staticmethod
    def clean_text(s: str) -> str:
        s = s.replace("\u200c", " ").replace("‌", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s


class LoadVoice:
    def __init__(self, rec):
        self.rec = rec

    def load_audio(self):
        audio, sr = sf.read(self.rec["audio"], dtype="float32")
        if sr != 16000:
            raise ValueError(f"Expected sample rate 16000, but got {sr}")

        s = int(round(float(self.rec["start"]) * sr))
        e = int(round(float(self.rec["end"])   * sr))

        n = len(audio)
        s = max(0, min(n, s))
        e = max(0, min(n, e))

        if e <= s:
            e = min(n, s + 10 * sr)
            if e <= s:
                raise ValueError("Invalid segment: end <= start and no room for fallback")
        if audio.ndim == 2:  
            audio = audio.mean(axis=1)

        return audio[s:e]
class ManifestBuilder:
    def check_manifest(self) -> None:
        if MANIFEST.exists():
            print(f"[**info**] Using existing manifest: {MANIFEST}")
            raise SystemExit(0)

        try:
            import pysrt  # noqa: F401
        except ImportError:
            raise ImportError("Please install pysrt package: pip install pysrt")

        if not AUDIO_DIR.exists() or not SUB_DIR.exists():
            raise FileNotFoundError("[**Error Not Found**] : Audio or Subtitle directory is missing.")

    def map_builder(self) -> None:
        # 1) build srt_map: basename -> Path(srt)
        srt_map: Dict[str, Path] = {}
        for srt in SUB_DIR.glob("*.srt"):
            name = srt.name
            base = None
            for suf in STR_SUFFIX:
                if name.endswith(suf):
                    base = name[: -len(suf)]
                    break
            if base is None:
                base = srt.stem
            srt_map[base] = srt

        # 2) pair wav <-> srt by basename
        pairs = []
        for wav in AUDIO_DIR.glob("*.wav"):
            base = wav.stem.replace("_16k", "")
            if base in srt_map:
                pairs.append((base, wav, srt_map[base]))
            else:
                print(f"[**Warning**] No matching subtitle for audio: {wav}")

        if not pairs:
            raise SystemExit("[**Error**] Please pair your audio and subtitle names correctly.")

        # 3) write manifest
        n = 0
        with MANIFEST.open("w", encoding="utf-8") as fout:
            for base, wav, srt in pairs:
                subs = pysrt.open(str(srt), encoding="utf-8")
                for i, item in enumerate(subs):
                    text = SRTPreprocessor.clean_text(item.text.replace("\n", " "))
                    if not text:
                        continue
                    start = SRTPreprocessor.srt_time_to_sec(item.start)
                    end = SRTPreprocessor.srt_time_to_sec(item.end)
                    dur = max(0.0, end - start)
                    if dur < 0.5 or dur > 30.5:
                        continue
                    rec = {
                        "uid": f"{base}-seg-{i:04d}",
                        "audio": str(wav).replace("\\", "/"),
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "text": text,
                        "language": LANG,
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1

        print(f"[ok] Wrote {n} segments -> {MANIFEST}")