from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
# Add local "src" directory to Python path so we can import asr.*
sys.path.append(str(ROOT / "src"))

import soundfile as sf
import torch as T
import pysrt
import evaluate

from asr.normalization.fa import FaNormalization
from asr.model.builder import ModelBuilderProcessor
from asr.model.base_builder import BaseModelBuilder
from asr.eval.segment_evaluator import WhisperSegmentEvaluator
from asr.configs.main_conf import LANG, TASK


class WhisperTranscriber:
    """
    Compare a fine-tuned Whisper model with the base Whisper model on:
      - the same audio segment
      - the same SRT reference window
    and compute WER/CER for both.
    """

    def __init__(self):
        # Select device (GPU if available, otherwise CPU)
        self.device = "cuda" if T.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {self.device}")

        # -------- Load fine-tuned model -------- #
        print("[INFO] Loading fine-tuned model...")
        ft_model, ft_processor = ModelBuilderProcessor.build_model(mode="inference")
        self.ft_model = ft_model.to(self.device)
        self.ft_processor = ft_processor
        self.ft_model.eval()  # disable dropout, etc.

        # Configure generation for fine-tuned model
        gc_ft = self.ft_model.generation_config
        gc_ft.language = LANG
        gc_ft.task = TASK
        gc_ft.do_sample = False
        gc_ft.num_beams = 1
        gc_ft.no_repeat_ngram_size = 4
        gc_ft.repetition_penalty = 1.1
        gc_ft.max_new_tokens = 225

        # -------- Load base (non-fine-tuned) model -------- #
        print("[INFO] Loading base model...")
        self.base_model, self.base_processor = BaseModelBuilder.build_base_model(
            device=self.device,
            eval_mode=True,
        )

        # -------- Metrics and normalization -------- #
        self._cer_metric = evaluate.load("cer")
        self._wer_metric = evaluate.load("wer")

        # Optional Farsi normalization (for fair metric computation)
        try:
            self._norm = FaNormalization()
        except Exception:
            self._norm = None
            print(
                "[WARN] FaNormalization could not be initialized; metrics will use raw text."
            )

    # -------------------- Audio helpers -------------------- #
    def _load_audio(self, wav_path: Path):
        """
        Load audio from disk using soundfile, ensure:
          - file exists
          - mono waveform
          - 16 kHz sample rate
        """
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")

        # Read as float32 for Whisper
        audio, sr = sf.read(str(wav_path), dtype="float32")

        # Convert stereo → mono by averaging channels
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        # Whisper expects 16 kHz audio
        if sr != 16000:
            raise ValueError(f"Audio must be 16kHz; got {sr}")

        return audio, sr

    def _features_from_audio(self, audio, sr, processor):
        """
        Convert a raw audio waveform into Whisper input features
        using the given processor (fine-tuned or base).
        """
        feat = processor.feature_extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
        )
        # Move features to the correct device (CPU / GPU)
        return feat["input_features"].to(self.device)

    # -------------------- SRT helper (time window) -------------------- #
    def _load_srt_window(
        self,
        srt_path: Path,
        start_sec: float = 0.0,
        duration_sec: float = 60.0,
    ) -> str:
        """
        Load a time window from an SRT file and concatenate all subtitle lines
        that overlap with [start_sec, start_sec + duration_sec).
        """
        if not srt_path.exists():
            raise FileNotFoundError(f"SRT file not found: {srt_path}")

        subs = pysrt.open(str(srt_path), encoding="utf-8")
        t0 = start_sec
        t1 = start_sec + duration_sec

        parts = []
        for s in subs:
            # pysrt stores times in milliseconds -> convert to seconds
            st = s.start.ordinal / 1000.0
            et = s.end.ordinal / 1000.0

            # Skip subtitles completely before the window
            if et <= t0:
                continue
            # Stop once subtitles start after the window
            if st >= t1:
                continue

            # Replace internal newlines with spaces
            txt = s.text.replace("\n", " ")
            parts.append(txt)

        # Concatenate all texts in the time window
        return " ".join(parts)

    # -------------------- Normalization + metrics -------------------- #
    def _norm_for_metric(self, s: str) -> str:
        """
        Normalize text before metric computation (Farsi-specific if available).
        """
        if self._norm is None:
            return s or ""
        return self._norm.for_metric(s or "")

    def _compute_metrics_pair(self, hyp: str, ref: str):
        """
        Compute CER and WER between hypothesis and reference.
        Both are normalized (if FaNormalization is available).
        Returns:
          cer, wer, hyp_norm, ref_norm
        """
        hyp_norm = self._norm_for_metric(hyp)
        ref_norm = self._norm_for_metric(ref)

        cer = self._cer_metric.compute(predictions=[hyp_norm], references=[ref_norm])
        wer = self._wer_metric.compute(predictions=[hyp_norm], references=[ref_norm])

        return cer, wer, hyp_norm, ref_norm

    # -------------------- Core transcription (FT + base) -------------------- #
    def transcribe_segment_both(
        self,
        wav_path: Path,
        start_sec: float = 0.0,
        duration_sec: float = 60.0,
    ):
        """
        Transcribe a given time window [start_sec, start_sec + duration_sec)
        from one audio file using:
          - fine-tuned model
          - base model
        Returns:
          hyp_ft, hyp_base
        """
        # Load full audio and sample rate
        audio, sr = self._load_audio(wav_path)
        n = len(audio)

        # Convert time (seconds) to sample indices
        s = max(0, int(round(start_sec * sr)))
        e = min(n, s + int(round(duration_sec * sr)))
        if s >= n:
            # Start beyond audio length -> empty output
            return "", ""

        # Slice the desired segment
        audio_seg = audio[s:e]

        # Extract features for both models (fine-tuned and base)
        ft_feats = self._features_from_audio(audio_seg, sr, self.ft_processor)
        base_feats = self._features_from_audio(audio_seg, sr, self.base_processor)

        # Disable gradients for inference
        with T.no_grad():
            # Fine-tuned model generation
            ft_ids = self.ft_model.generate(
                input_features=ft_feats,
                language=LANG,
                task=TASK,
                do_sample=False,
                num_beams=self.ft_model.generation_config.num_beams,
                max_new_tokens=self.ft_model.generation_config.max_new_tokens,
            )

            # Base model generation
            base_ids = self.base_model.generate(
                input_features=base_feats,
                language=LANG,
                task=TASK,
                do_sample=False,
                num_beams=self.base_model.generation_config.num_beams,
                max_new_tokens=self.base_model.generation_config.max_new_tokens,
            )

        # Decode token IDs back to text for both models
        hyp_ft = self.ft_processor.tokenizer.batch_decode(
            ft_ids, skip_special_tokens=True
        )[0]

        hyp_base = self.base_processor.tokenizer.batch_decode(
            base_ids, skip_special_tokens=True
        )[0]

        return hyp_ft, hyp_base


if __name__ == "__main__":
    # Example single-file evaluation:
    #  - 60 seconds from a processed 16kHz WAV
    #  - matching SRT file in the same language
    AUDIO = Path("audio/processed/test5_16k.wav")
    SRT = Path("subtitle/test5.fa.srt")

    print("[DEBUG] AUDIO:", AUDIO)
    print("[DEBUG] SRT  :", SRT)

    OUT_TXT = Path("runs/final_output.txt")

    app = WhisperTranscriber()
    evaluator = WhisperSegmentEvaluator(app)

    evaluator.evaluate_single_with_srt(
        AUDIO,
        SRT,
        OUT_TXT,
        start_sec=0.0,
        duration_sec=60.0,
    )
