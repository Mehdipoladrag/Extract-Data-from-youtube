
from typing import Any, Dict, List, Optional
import soundfile as sf


class LoadVoice:
    """Safe audio loading and segment slicing using start/end (falls back to full audio if missing)."""

    def __init__(self, rec: Dict[str, Any]):
        self.rec = rec

    def load_audio(self):
        audio, sr = sf.read(self.rec["audio"], dtype="float32")
        if sr != 16000:
            raise ValueError(
                f"Expected sample rate 16000, got {sr} for {self.rec.get('audio')}"
            )
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        n = len(audio)
        try:
            s = int(round(float(self.rec["start"]) * sr))
            e = int(round(float(self.rec["end"]) * sr))
        except Exception:
            return audio, sr

        s = max(0, min(n, s))
        e = max(0, min(n, e))
        if s >= n:
            tail = min(10 * sr, n)
            s = n - tail
            e = n
        if e <= s:
            e = min(n, s + 10 * sr)
            if e - s < 1 * sr:
                s = max(0, min(s, n - 1 * sr))
                e = min(n, s + 1 * sr)
            if e <= s:
                s, e = 0, n
        return audio[s:e], sr
