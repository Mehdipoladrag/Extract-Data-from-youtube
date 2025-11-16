import json
import pysrt
from typing import Any, Dict, List, Optional
from pathlib import Path
from asr.configs.main_conf import LANG, MANIFEST, AUDIO_DIR, SUB_DIR, STR_SUFFIX
from asr.normalization.srt import SRTPreprocessor


class ManifestBuilder:
    """Pair .wav with .srt and write JSONL manifest; skips build if manifest already exists."""

    def build_if_needed(self) -> None:
        if MANIFEST.exists():
            print(f"[info] Using existing manifest: {MANIFEST}")
            return
        if not AUDIO_DIR.exists() or not SUB_DIR.exists():
            raise FileNotFoundError("[Error] Audio or Subtitle directory is missing.")
        self._map_builder()

    def _map_builder(self) -> None:
        srt_map: Dict[str, Path] = {}
        for srt in SUB_DIR.glob("*.srt"):
            base = None
            for suf in STR_SUFFIX:
                if srt.name.endswith(suf):
                    base = srt.name[: -len(suf)]
                    break
            if base is None:
                base = srt.stem
            srt_map[base] = srt

        pairs = []
        for wav in AUDIO_DIR.glob("*.wav"):
            base = wav.stem.replace("_16k", "")
            if base in srt_map:
                pairs.append((base, wav, srt_map[base]))
            else:
                print(f"[warn] No matching subtitle for audio: {wav}")

        if not pairs:
            raise SystemExit(
                "[Error] Please pair your audio and subtitle names correctly."
            )

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