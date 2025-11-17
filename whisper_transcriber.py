from pathlib import Path
from pathlib import Path
import sys
import soundfile as sf
import torch as T
import pysrt
import evaluate
from src.asr.normalization.fa import FaNormalization
from asr.model.builder import ModelBuilderProcessor
from asr.configs.main_conf import LANG, TASK

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

class WhisperTranscriber:
    
    def __init__(self):
        model, processor = ModelBuilderProcessor.build_model()
        self.model = model
        self.processor = processor

        self.device = "cuda" if T.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        # generation config
        gc = self.model.generation_config
        gc.language = LANG
        gc.task = TASK
        gc.do_sample = False
        gc.num_beams = 1
        gc.no_repeat_ngram_size = 4
        gc.repetition_penalty = 1.1
        gc.max_new_tokens = 225

   
        self._cer_metric = evaluate.load("cer")
        self._wer_metric = evaluate.load("wer")
        try:
            self._norm = FaNormalization()
        except Exception:
            self._norm = None 

    # -------------------- Audio helpers -------------------- #
    def _load_audio(self, wav_path: Path):
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != 16000:
            raise ValueError(f"Audio must be 16kHz; got {sr}")
        return audio, sr

    def _features_from_audio(self, audio, sr):
        feat = self.processor.feature_extractor(
            audio, sampling_rate=sr, return_tensors="pt"
        )
        return feat["input_features"].to(self.device)

    # -------------------- SRT helper (window) -------------------- #
    def _load_srt_window(
        self,
        srt_path: Path,
        start_sec: float = 0.0,
        duration_sec: float = 60.0,
    ) -> str:
        
        if not srt_path.exists():
            raise FileNotFoundError(f"SRT file not found: {srt_path}")

        subs = pysrt.open(str(srt_path), encoding="utf-8")
        t0 = start_sec
        t1 = start_sec + duration_sec

        parts = []
        for s in subs:
            st = s.start.ordinal / 1000.0  
            et = s.end.ordinal / 1000.0    

            if et <= t0:
                continue
            if st >= t1:
                continue

            txt = s.text.replace("\n", " ")
            parts.append(txt)

        return " ".join(parts)

    # -------------------- Normalization helper -------------------- #
    def _norm_for_metric(self, s: str) -> str:
        
        if self._norm is None:
            return s

        fn = getattr(self._norm, "for_metric", None)
        if callable(fn):
            return fn(s)

        for name in ("normalize", "normalize_for_metric", "for_eval", "__call__"):
            meth = getattr(self._norm, name, None)
            if callable(meth):
                try:
                    return meth(s)
                except Exception:
                    pass

        return s

    # -------------------- Metrics -------------------- #
    def _compute_metrics_pair(self, hyp: str, ref: str):
        
        hyp_norm = self._norm_for_metric(hyp)
        ref_norm = self._norm_for_metric(ref)

        cer = self._cer_metric.compute(
            predictions=[hyp_norm], references=[ref_norm]
        )
        wer = self._wer_metric.compute(
            predictions=[hyp_norm], references=[ref_norm]
        )
        return cer, wer, hyp_norm, ref_norm

    # -------------------- Core transcription (segment) -------------------- #
    def transcribe_segment(
        self,
        wav_path: Path,
        start_sec: float = 0.0,
        duration_sec: float = 60.0,
    ) -> str:
        
        audio, sr = self._load_audio(wav_path)
        n = len(audio)

        s = max(0, int(round(start_sec * sr)))
        e = min(n, s + int(round(duration_sec * sr)))
        if s >= n:
            return ""

        audio_seg = audio[s:e]
        input_features = self._features_from_audio(audio_seg, sr)

        with T.no_grad():
            pred_ids = self.model.generate(
                input_features=input_features,
                language=LANG,
                task=TASK,
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

    # -------------------- EVAL: audio segment + srt window → txt -------------------- #
    def evaluate_single_with_srt(
        self,
        audio_path: Path,
        srt_path: Path,
        out_txt: Path,
        start_sec: float = 0.0,
        duration_sec: float = 60.0,
    ):
        
        print(f"[INFO] Transcribing: {audio_path} (from {start_sec}s for {duration_sec}s)")
        hyp = self.transcribe_segment(audio_path, start_sec, duration_sec)

        print(f"[INFO] Loading SRT: {srt_path} (same time window)")
        ref = self._load_srt_window(srt_path, start_sec, duration_sec)

        cer, wer, hyp_norm, ref_norm = self._compute_metrics_pair(hyp, ref)

        out_txt.parent.mkdir(parents=True, exist_ok=True)
        with out_txt.open("w", encoding="utf-8") as f:
            f.write("AUDIO FILE:\n")
            f.write(str(audio_path) + "\n\n")
            f.write("SRT FILE:\n")
            f.write(str(srt_path) + "\n\n")
            f.write(f"TIME WINDOW: [{start_sec:.1f}s, {start_sec+duration_sec:.1f}s)\n\n")

            f.write("===== RAW TEXTS =====\n")
            f.write("REF (from SRT window):\n")
            f.write(ref + "\n\n")
            f.write("HYP (model output):\n")
            f.write(hyp + "\n\n")

            f.write("===== NORMALIZED =====\n")
            f.write("REF_NORM:\n")
            f.write(ref_norm + "\n\n")
            f.write("HYP_NORM:\n")
            f.write(hyp_norm + "\n\n")

            f.write("===== METRICS (normalized, time window) =====\n")
            f.write(f"WER: {wer:.6f}\n")
            f.write(f"CER: {cer:.6f}\n")

        print("\n=== EVAL DONE (segment) ===")
        print(f"WER: {wer:.6f}  |  CER: {cer:.6f}")
        print(f"Saved detailed results to: {out_txt}")


# ======================= MAIN ======================= #

if __name__ == "__main__":
    AUDIO = Path(
        "audio/processed/test5_16k.wav"
    )
    SRT = Path(
        "subtitle/test5.fa.srt"
    )

    print("[DEBUG] AUDIO:", AUDIO)
    print("[DEBUG] SRT  :", SRT)

    OUT_TXT = Path("runs/whisper_single_eval_60s.txt")

    app = WhisperTranscriber()
    app.evaluate_single_with_srt(
        AUDIO,
        SRT,
        OUT_TXT,
        start_sec=0.0,
        duration_sec=60.0,  
    )
