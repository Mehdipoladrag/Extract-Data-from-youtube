import torch as T
from pathlib import Path
import random
import numpy as np

T.backends.cuda.matmul.fp32_precision = "tf32"
LANG = "fa"
TASK = "transcribe"
MODEL_NAME = "openai/whisper-small"
SEED = 42
MANIFEST = Path("manifest.jsonl")
AUDIO_DIR = Path("audio/processed")
SUB_DIR = Path("subtitle")
STR_SUFFIX = [".fa.srt", ".srt"]

random.seed(SEED)
np.random.seed(SEED)
T.manual_seed(SEED)