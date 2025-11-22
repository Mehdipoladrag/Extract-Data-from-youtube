from pathlib import Path
from typing import Union


class WhisperSegmentEvaluator:
    """
    Evaluate a single audio + SRT pair on a specific time window using
    a given transcriber (fine-tuned + base Whisper models).

    The transcriber instance must implement:
      - transcribe_segment_both(audio_path, start_sec, duration_sec) -> (hyp_ft, hyp_base)
      - _load_srt_window(srt_path, start_sec, duration_sec) -> reference text
      - _compute_metrics_pair(hyp, ref) -> (cer, wer, hyp_norm, ref_norm)
    """

    def __init__(self, transcriber: object):
        # Keep a reference to the transcriber (WhisperTranscriber)
        self.transcriber = transcriber

    def evaluate_single_with_srt(
        self,
        audio_path: Union[Path, str],
        srt_path: Union[Path, str],
        out_txt: Union[Path, str],
        start_sec: float = 0.0,
        duration_sec: float = 60.0,
    ):
        """
        Evaluate one audio file + SRT pair on a given time window:
          - transcribe with fine-tuned model and base model
          - extract reference text from SRT in same time range
          - compute WER/CER for both models
          - write a detailed report to out_txt
        """
        audio_path = Path(audio_path)
        srt_path = Path(srt_path)
        out_txt = Path(out_txt)

        print(
            f"[INFO] Transcribing (FT + BASE): {audio_path} "
            f"(from {start_sec}s for {duration_sec}s)"
        )

        # Hypotheses from both models (fine-tuned + base)
        hyp_ft, hyp_base = self.transcriber.transcribe_segment_both(
            audio_path, start_sec, duration_sec
        )

        print(f"[INFO] Loading SRT: {srt_path} (same time window)")
        # Reference text from SRT in the same time window
        ref = self.transcriber._load_srt_window(srt_path, start_sec, duration_sec)

        # Compute metrics for fine-tuned model
        cer_ft, wer_ft, hyp_ft_norm, ref_norm = self.transcriber._compute_metrics_pair(
            hyp_ft, ref
        )
        # Compute metrics for base model
        cer_base, wer_base, hyp_base_norm, _ = self.transcriber._compute_metrics_pair(
            hyp_base, ref
        )

        # Ensure output directory exists
        out_txt.parent.mkdir(parents=True, exist_ok=True)

        # Write detailed comparison report to text file
        with out_txt.open("w", encoding="utf-8") as f:
            f.write("AUDIO FILE:\n")
            f.write(str(audio_path) + "\n\n")

            f.write("SRT FILE:\n")
            f.write(str(srt_path) + "\n\n")

            f.write(
                f"TIME WINDOW: [{start_sec:.1f}s, {start_sec+duration_sec:.1f}s)\n\n"
            )

            f.write("===== RAW TEXTS =====\n")
            f.write("REF (from SRT window):\n")
            f.write(ref + "\n\n")

            f.write("HYP_FT (fine-tuned model):\n")
            f.write(hyp_ft + "\n\n")

            f.write("HYP_BASE (base model):\n")
            f.write(hyp_base + "\n\n")

            f.write("===== NORMALIZED =====\n")
            f.write("REF_NORM:\n")
            f.write(ref_norm + "\n\n")

            f.write("HYP_FT_NORM:\n")
            f.write(hyp_ft_norm + "\n\n")

            f.write("HYP_BASE_NORM:\n")
            f.write(hyp_base_norm + "\n\n")

            f.write("===== METRICS (normalized, time window) =====\n")
            f.write(f"[FT]   WER: {wer_ft:.6f} | CER: {cer_ft:.6f}\n")
            f.write(f"[BASE] WER: {wer_base:.6f} | CER: {cer_base:.6f}\n")

        print("\n=== EVAL DONE (segment) ===")
        print(f"[FT]   WER: {wer_ft:.6f} | CER: {cer_ft:.6f}")
        print(f"[BASE] WER: {wer_base:.6f} | CER: {cer_base:.6f}")
        print(f"Saved detailed results to: {out_txt}")
