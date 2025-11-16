# -*- coding: utf-8 -*-
from pathlib import Path

import soundfile as sf
import torch as T
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class WhisperTranscriber:
    """Minimal Whisper transcriber with chunking."""

    def __init__(self, model_name: str = "openai/whisper-small", language: str = "fa"):
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

        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()

        forced = self.processor.get_decoder_prompt_ids(language=language, task="transcribe")
        self.model.generation_config.forced_decoder_ids = forced
        gc = self.model.generation_config
        gc.do_sample = False
        gc.num_beams = 5        
        gc.no_repeat_ngram_size = 3
        gc.repetition_penalty = 1.0
        gc.max_new_tokens = 225

    def _load_audio(self, wav_path: Path):
        """Read WAV, mono float32 @16kHz."""
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != 16000:
            raise ValueError(f"Audio must be 16kHz; got {sr}")
        return audio, sr

    def _features_from_audio(self, audio, sr):
        """Compute input_features tensor on device."""
        feat = self.processor.feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
        return feat["input_features"].to(self.device)

    def _decode_ids(self, pred_ids):
        """Token IDs -> text."""
        try:
            pred_ids = pred_ids.detach().cpu()
        except Exception:
            pass
        out = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        return out[0].strip() if out else ""

    def transcribe(self, wav_path: Path, start_sec: float = 0.0, duration_sec: float = 60.0,
                   chunk_s: float = 30.0, overlap_s: float = 4.0) -> str:
        """Transcribe [start, start+duration] via 30s chunks with small overlap."""
        audio, sr = self._load_audio(wav_path)

        n = len(audio)
        s_global = max(0, int(round(start_sec * sr)))
        e_global = min(n, s_global + int(round(duration_sec * sr)))
        if s_global >= n or e_global <= s_global:
            return ""
        audio = audio[s_global:e_global]
        n = len(audio)

        rms = float((audio**2).mean() ** 0.5)
        if rms > 1e-6:
            target = 0.1
            audio = audio * (target / rms)

        # chunking
        win = int(chunk_s * sr)
        hop = win - int(overlap_s * sr) if overlap_s < chunk_s else win
        texts, i = [], 0

        with T.no_grad():
            while i < n:
                j = min(n, i + win)
                seg = audio[i:j]
                if seg.size == 0:
                    break

                feats = self._features_from_audio(seg, sr)
                ids = self.model.generate(
                    feats,
                    do_sample=False,
                    num_beams=self.model.generation_config.num_beams,
                    no_repeat_ngram_size=self.model.generation_config.no_repeat_ngram_size,
                    repetition_penalty=self.model.generation_config.repetition_penalty,
                    max_new_tokens=self.model.generation_config.max_new_tokens,
                )
                texts.append(self._decode_ids(ids))

                if hop <= 0:
                    break
                nxt = i + hop
                if nxt + win > n and (n - win) > i:
                    i = n - win
                else:
                    i = nxt

                if win >= n:  # very short clips
                    break

        return " ".join(t for t in (x.strip() for x in texts) if t)

if __name__ == "__main__":
    MODEL = "openai/whisper-small"
    LANG = "fa"
    AUDIO = Path("audio/processed/چطور کاریزماتیک باشیم؟ راز آدمای خاص!_16k.wav")

    app = WhisperTranscriber(model_name=MODEL, language=LANG)

    text_60s = app.transcribe(AUDIO, start_sec=0, duration_sec=60)
    print("\n--- 60s TRANSCRIPT ---\n", text_60s)

    