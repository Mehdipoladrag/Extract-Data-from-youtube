# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import noisereduce as nr


class NoiseReducer:
   

    def __init__(
        self,
        stationary: bool = True,
        prop_decrease: float = 0.9,
    ) -> None:
        self.stationary = stationary
        self.prop_decrease = prop_decrease

    # ---------- helpers ---------- #
    def load(self, path: Path) -> Tuple[np.ndarray, int]:
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        audio, sr = sf.read(str(path), dtype="float32")

        # stereo → mono
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        return audio, sr

    def save(self, path: Path, audio: np.ndarray, sr: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio, sr)

    def clean_array(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        y = audio.astype("float32")

        y_reduced = nr.reduce_noise(
            y=y,
            sr=sr,
            stationary=self.stationary,
            prop_decrease=self.prop_decrease,
        )
        return y_reduced.astype("float32"), sr

    def clean_file(self, in_path: Path, out_path: Path) -> Path:
        
        audio, sr = self.load(in_path)
        clean, sr = self.clean_array(audio, sr)

        self.save(out_path, clean, sr)
        return out_path
