import os, json, random, re
import torch as T
import pysrt
import numpy as np
import soundfile as sf
import evaluate
from pathlib import Path
from typing import List, Dict, Any, Optional
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
TASK = "transcribe"
MODEL_NAME = "openai/whisper-small"
SEED = 42
MANIFEST = Path("manifest.json")
AUDIO_DIR = Path("audi/processed")
SUB_DIR = Path("subtitle")
STR_SUFFIX = [".fa.srt", ".srt"]

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


class SRTPreprocessor:
    """
    A utility class for preprocessing subtitle (.srt) files.

    Main purposes:
    1. Convert SRT timestamp objects (hours, minutes, seconds, milliseconds)
       into a single floating-point value representing total seconds.
       This is useful for aligning subtitle text with audio segments.

    2. Clean and normalize subtitle text:
       - Replace inconsistent zero-width and regular spaces.
       - Remove extra whitespace and line breaks.
       - Produce a clean, standardized string ready for training or evaluation
         in speech-to-text models such as Whisper.
    """
    @staticmethod
    def srt_time_to_sec(t) -> float:
        return t.hours * 3600 + t.minutes * 60 + t.seconds + t.milliseconds / 1000.0

    @staticmethod
    def clean_text(s: str) -> str:
        s = s.replace("\u200c", " ").replace("‌", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s


class LoadVoice:
    """
    A utility class for loading and slicing audio segments based on
    start and end timestamps (in seconds) defined in a record.

    Main purposes:
    1. Load the corresponding audio file for a given data record (`rec`).
    2. Verify that the audio has a 16kHz sample rate (required for Whisper).
    3. Convert the `start` and `end` times (in seconds) into sample indices.
    4. Safely slice the audio array to extract the desired segment.
       - Ensures indices stay within valid audio bounds.
       - Falls back to a default 10-second segment if timestamps are invalid.
    5. If the audio is stereo (2 channels), convert it to mono by averaging.

    Returns:
        numpy.ndarray (float32): The extracted mono audio segment.
    """
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
    """
    A utility class responsible for creating the dataset manifest file (manifest.jsonl)
    used for Whisper fine-tuning or speech-to-text training.

    Main responsibilities:
    1. Validate dataset structure:
       - Check if the manifest already exists and stop if so.
       - Ensure required directories (audio/processed and subtitle) exist.
       - Verify that the `pysrt` library is installed for subtitle parsing.

    2. Build a mapping between audio (.wav) files and their corresponding subtitle (.srt) files
       based on matching base filenames (e.g., "clip01.wav" <-> "clip01.srt").

    3. Parse SRT files using `pysrt` to extract:
       - Cleaned subtitle text (via SRTPreprocessor.clean_text).
       - Start and end times (converted to seconds).
       - Language information.

    4. Write the extracted data into a JSONL (line-by-line JSON) manifest file.
       Each entry includes:
           {
               "uid": "file-segment-id",
               "audio": "path/to/audio.wav",
               "start": float (seconds),
               "end": float (seconds),
               "text": "cleaned transcription text",
               "language": LANG
           }

    The resulting manifest file is essential for building datasets that align
    audio segments with their corresponding text, enabling model fine-tuning.
    """
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
class ModelBuilderProcessor:
    """
    A utility class for initializing the Whisper model and processor with LoRA configuration.

    Responsibilities:
    1. Load the WhisperProcessor and WhisperForConditionalGeneration models from Hugging Face.
    2. Ensure the tokenizer has a valid pad_token (defaults to eos_token if missing).
    3. Set language and task constraints (forced_decoder_ids) for consistent generation.
    4. Apply a PEFT LoRA (Low-Rank Adaptation) configuration to enable lightweight fine-tuning.
    5. Return both the model and processor, ready for training or inference.

    Returns:
        Tuple[WhisperForConditionalGeneration, WhisperProcessor]
    """

    @staticmethod
    def build_model():
        processor = WhisperProcessor.from_pretrained(MODEL_NAME)

        # Ensure pad_token exists (some Whisper tokenizers lack one)
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

        # Load Whisper base model
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

        # Lock language and task configuration
        forced_ids = processor.get_decoder_prompt_ids(language=LANG, task=TASK)
        model.generation_config.forced_decoder_ids = forced_ids
        model.config.forced_decoder_ids = forced_ids

        # Configure LoRA (low-rank fine-tuning) for efficiency
        peft_cfg = LoraConfig(
            task_type="SEQ_2_SEQ_LM",
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        )

        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()  # Debug info

        return model, processor

class MapExampler:
    """
    A class that prepares individual dataset examples for Whisper fine-tuning.

    Responsibilities:
    1. Load a single audio segment from disk using LoadVoice.
    2. Convert the waveform to Whisper input features (log-Mel spectrograms).
    3. Tokenize the reference transcription text as target labels.
    4. Return both `input_features` and `labels` ready for model training.

    Methods:
        mapping(rec: Dict) -> Dict[str, np.ndarray | List[int]]
            Process a dataset record into model-ready format.

    Returns:
        {
            "input_features": np.ndarray of shape (80, T),
            "labels": List[int] (token IDs for transcription)
        }
    """

    def __init__(self, processor: WhisperProcessor):
        """Initialize with a shared WhisperProcessor for feature extraction and tokenization."""
        self.processor = processor

    def mapping(self, rec):
        """Convert one record (audio + text) into model input and label tensors."""
        # Load and crop the audio segment
        audio, sr = LoadVoice(rec).load_audio()

        # Convert raw audio to Whisper log-Mel features
        x = self.processor.feature_extractor(
            audio, sampling_rate=sr, return_tensors="np"
        )
        input_features = x["input_features"][0]

        # Tokenize target text
        with self.processor.as_target_processor():
            y = self.processor(
                rec["text"], add_special_tokens=True, return_tensors=None
            )
        labels = y["input_ids"]

        return {
            "input_features": input_features,
            "labels": labels,
        }



class DataCollatorSpeechSeq2Seq:
    """
    OOP collator for Whisper-style speech seq2seq training.
    - Pads log-mel features along time axis to the max length in the batch.
    - Pads label sequences with tokenizer.pad_token_id (or eos) then replaces with -100.
    """

    def __init__(
        self,
        tokenizer: Any,
        feature_dim: int = 80,
        pad_to_multiple_of: Optional[int] = None, 
    ) -> None:
        self.tokenizer = tokenizer
        self.feature_dim = feature_dim
        self.pad_to_multiple_of = pad_to_multiple_of

        self._pad_id = (
            tokenizer.pad_token_id
            if getattr(tokenizer, "pad_token_id", None) is not None
            else tokenizer.eos_token_id
        )

    # --------- public API ---------
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, T.Tensor]:
        feats = [self._prepare_feature(f["input_features"]) for f in features]
        input_features = self._pad_features(feats)

        labels = [self._prepare_labels(f["labels"]) for f in features]
        labels_padded = self._pad_labels(labels)

        return {"input_features": input_features, "labels": labels_padded}

    # --------- feature helpers ---------
    def _prepare_feature(self, x: Any) -> T.Tensor:
        # list -> np -> tensor(float32)
        if isinstance(x, list):
            x = np.asarray(x, dtype=np.float32)
        if isinstance(x, np.ndarray):
            x = T.tensor(x, dtype=T.float32)
        if not isinstance(x, T.Tensor):
            x = T.tensor(x, dtype=T.float32)

        # squeeze potential batch dim (1, 80, T) or (1, T) -> handle robustly
        if x.ndim == 3 and x.shape[0] == 1:
            x = x.squeeze(0)

        # Validate shape: expect (feature_dim, time)
        if x.ndim == 1:
            raise ValueError(f"input_features is 1D {tuple(x.shape)}; expected (feature_dim, time).")
        if x.ndim != 2:
            raise ValueError(f"Unexpected input_features ndim={x.ndim}, shape={tuple(x.shape)}")
        if x.shape[0] != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {x.shape[0]} with shape {tuple(x.shape)}")

        return x

    def _pad_features(self, feats: List[T.Tensor]) -> T.Tensor:
        # time lengths
        times = [x.shape[1] for x in feats]
        max_time = max(times)

        if self.pad_to_multiple_of:
            remainder = max_time % self.pad_to_multiple_of
            if remainder:
                max_time += self.pad_to_multiple_of - remainder

        padded = []
        for x in feats:
            pad_len = max_time - x.shape[1]
            if pad_len > 0:
                pad = T.zeros((x.shape[0], pad_len), dtype=x.dtype, device=x.device)
                x = T.cat([x, pad], dim=1)
            padded.append(x)

        # Stack to (batch, feature_dim, time)
        return T.stack(padded, dim=0)

    # --------- label helpers ---------
    def _prepare_labels(self, ids: Any) -> T.Tensor:
        if isinstance(ids, list):
            return T.tensor(ids, dtype=T.long)
        if isinstance(ids, np.ndarray):
            return T.from_numpy(ids.astype(np.int64))
        if isinstance(ids, T.Tensor):
            return ids.to(dtype=T.long)
        # last resort
        return T.tensor(list(ids), dtype=T.long)

    def _pad_labels(self, labels: List[T.Tensor]) -> T.Tensor:
        padded = T.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self._pad_id
        )
        # Replace padding with -100 for loss ignore
        padded = padded.masked_fill(padded == self._pad_id, -100)
        return padded
