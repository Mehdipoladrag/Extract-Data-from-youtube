# clean_dataset_with_nr.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from src.asr.normalization.audio_clean.audio_clean import NoiseReducer


if __name__ == "__main__":
    in_dir = ROOT / "audio2/wav_audio"
    out_dir = ROOT / "audio_clean"

    cleaner = NoiseReducer(
        stationary=True,
        prop_decrease=0.9,
    )

    for wav_path in in_dir.rglob("*.wav"):
        rel = wav_path.relative_to(in_dir)
        out_path = out_dir / rel

        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[DENOISE] {wav_path} -> {out_path}")
        cleaner.clean_file(wav_path, out_path)
