# Whisper FA — Single‑File Inference & LoRA Fine‑Tuning

[![OpenAI Whisper][whisper-badge]][whisper]
[![Transformers][transformers-badge]][transformers]
[![PEFT / LoRA][peft-badge]][peft]
[![evaluate][evaluate-badge]][evaluate]
[![PyTorch][pytorch-badge]][pytorch]
[![CUDA][cuda-badge]][cuda]
[![Python][python-badge]][python]

Lightweight, production‑minded scripts to **transcribe Persian (fa) audio with OpenAI Whisper** and to **fine‑tune Whisper via LoRA** on your own audio+subtitle pairs.

- 🔊 **Inference**: one class (`WhisperTranscriber`) with stable generation settings  
- 🧠 **Training**: end‑to‑end LoRA pipeline (dataset prep → mapping → Trainer → save)  
- 🧩 **Data I/O**: simple JSONL manifest built from `.wav` + `.srt`  
- 🇮🇷 **Default language**: Persian (`fa`) out of the box (overrideable)  
- ⚡ **CUDA‑aware**: pins model once, moves tensors to device, enables TF32 where available  

---

## Table of Contents

- [Features](#features)
- [Repo Layout](#repo-layout)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
  - [1) Create environment](#1-create-environment)
  - [2) Install dependencies](#2-install-dependencies)
  - [3) Prepare data](#3-prepare-data)
  - [4) Run inference](#4-run-inference)
  - [5) Fine‑tune with LoRA](#5-fine-tune-with-lora)
- [Usage Details](#usage-details)
  - [Inference (`WhisperTranscriber`)](#inference-whispertranscriber)
  - [Training Pipeline (LoRA)](#training-pipeline-lora)
  - [Manifest format](#manifest-format)
  - [Subtitle pairing rules](#subtitle-pairing-rules)
- [Tips & Troubleshooting](#tips--troubleshooting)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Features

- **Single‑file inference** with [`WhisperProcessor`][whisper-doc] + [`WhisperForConditionalGeneration`][whisper-doc]  
- **Stable decoding** (greedy, no‑repeat n‑grams, repetition penalty, max length caps)  
- **Strict audio checks** (mono, **16 kHz**, `float32`)  
- **LoRA fine‑tuning** targeting attention & MLP modules (`q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`)  
- **Custom collator** for 80‑bin log‑mels (`(feature_dim, time)`), label masking to `-100`  
- **Metrics**: CER via [`evaluate`][evaluate]  
- **Trainer**: gradient checkpointing, TensorBoard, periodic eval/save  

---

## Repo Layout

```
.
├── README.md                  # this file
├── requirements.txt           # pinned versions (CUDA 12.8 build of torch)
├── whisper_transcribe.py      # WhisperTranscriber class (single-file inference)
├── fine_tuning.py             # full LoRA training pipeline
├── audio/
│   └── processed/             # 16 kHz mono WAV files (e.g., *_16k.wav)
├── subtitle/                  # SRT files (*.srt or *.fa.srt)
└── runs/
    └── whisper-fa-lora/       # training outputs (checkpoints, logs, final/)
```

> Make sure your script filenames are `whisper_transcribe.py` and `fine_tuning.py` to match the commands below.

---

## Requirements

- Python **3.10–3.12**  
- NVIDIA GPU (optional but recommended) with CUDA **12.1+** for the pinned `torch==2.9.0+cu128`  
- FFmpeg (for your own preprocessing, if needed)

Install dependencies from `requirements.txt` (pinned). If you only need inference, a minimal subset (`transformers`, `torch`, `soundfile`) is enough.

---

## Quickstart

### 1) Create environment

```bash
# using venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

If you hit CUDA/version issues, install a matching PyTorch from the official site first:  
<https://pytorch.org/get-started/locally/>

### 3) Prepare data

Place **16 kHz mono** WAV files in `audio/processed/` and matching SRT subtitles in `subtitle/`:

- Audio naming example: `MyLecture_16k.wav`  
- Subtitle naming (any of):  
  - `MyLecture.fa.srt`  
  - `MyLecture.srt`

### 4) Run inference

`whisper_transcribe.py` contains the `WhisperTranscriber`:

```bash
python whisper_transcribe.py
```

Default config:
- `model_name="openai/whisper-small"`  
- `language="fa"`  
- Example audio path inside the script (`audio/processed/..._16k.wav`)

Use in your own code:

```python
from pathlib import Path
from whisper_transcribe import WhisperTranscriber

app = WhisperTranscriber(model_name="openai/whisper-small", language="fa")
text = app.transcribe(Path("audio/processed/example_16k.wav"))
print(text)
```

### 5) Fine‑tune with LoRA

`fine_tuning.py` implements a full LoRA pipeline that:

1. Builds a JSONL manifest from your audio+SRT pairs (skips if `manifest.jsonl` exists)  
2. Maps each SRT segment to a training example (80×T features + tokenized labels)  
3. Trains with `Seq2SeqTrainer` (gradient checkpointing, TensorBoard, eval/save steps)  
4. Saves model + processor to `runs/whisper-fa-lora/final/`

Run:

```bash
python fine_tuning.py
```

Training defaults (see `WhisperTrainingPipeline` init args):
- `MODEL_NAME="openai/whisper-small"`  
- `LANG="fa"`, `TASK="transcribe"`  
- Steps: `max_steps=800`, `warmup_steps=50`  
- Batch sizes: `train=16`, `eval=8` (tune for your GPU)  
- Mixed precision: `fp16=True` (set `False` if unsupported)  
- Generation caps: `generation_max_length=225`, beams=1

TensorBoard:

```bash
tensorboard --logdir runs/whisper-fa-lora
```

---

## Usage Details

### Inference (`WhisperTranscriber`)

Key behavior:
- Loads `WhisperProcessor` & `WhisperForConditionalGeneration` once, moves to `cuda` if available  
- Enforces **16 kHz** mono `float32` input (raises if mismatched)  
- Stable decoding settings:  
  - greedy (`do_sample=False`, `num_beams=1`)  
  - `no_repeat_ngram_size=4`, `repetition_penalty=1.1`  
  - `max_new_tokens=225`  
- Passes `language` and `task="transcribe"` to both `generation_config` and `generate()`

Tips:
- If your audio isn’t 16 kHz mono, resample beforehand:  
  `ffmpeg -i in.wav -ar 16000 -ac 1 out_16k.wav`  
- Long files: The example shows a single‑pass call. For very long content, chunk externally and stitch text.

### Training Pipeline (LoRA)

Components:
- **`ManifestBuilder`**: pairs WAV↔SRT by base name; writes `manifest.jsonl` once  
- **`MapExampler`**: converts manifest record → `(input_features, labels)`  
- **`DataCollatorSpeechSeq2Seq`**: pads features to same time length; masks padded labels to `-100`  
- **`ComputeMetrics`**: CER via `evaluate`  
- **`WhisperSafeTrainer`**: feeds `input_features` directly into base model under PEFT; stable `generate()` during eval  
- **`WhisperTrainingPipeline`**: glues everything together

LoRA config (PEFT):
- `r=16`, `lora_alpha=32`, `lora_dropout=0.1`  
- Targets: `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`

Outputs:
- Checkpoints under `runs/whisper-fa-lora/`  
- Final artifacts saved to `runs/whisper-fa-lora/final/`

### Manifest format

`manifest.jsonl` (one JSON per line):

```json
{
  "uid": "MyLecture-seg-0000",
  "audio": "audio/processed/MyLecture_16k.wav",
  "start": 12.345,
  "end": 18.901,
  "text": "Normalized subtitle text here",
  "language": "fa"
}
```

### Subtitle pairing rules

- For each `audio/processed/<BASE>_16k.wav`, the builder looks for these endings in `subtitle/`:  
  - `<BASE>.fa.srt`  
  - `<BASE>.srt`  
- SRT segments are filtered by duration: **0.5s ≤ dur ≤ 30.5s**  
- Text is normalized (ZWNJ → space, collapse whitespace)  
- If a match is missing, you’ll see: `[warn] No matching subtitle for audio: ...`

---

## Tips & Troubleshooting

- **“Audio must be 16kHz”**: resample your file to 16 kHz mono.  
- **OOM during training**: lower `per_device_train_batch_size`, increase `gradient_accumulation_steps`, or reduce `max_steps`.  
- **Tokenizer pad warnings**: code sets `pad_token=eos_token` if absent.  
- **Slow eval/generate**: reduce `generation_max_length` from 225 if your segments are short.  
- **CPU‑only**: it works, but training will be slow. Consider `fp16=False` automatically when CUDA is unavailable.  
- **Version mismatches**: this repo pins versions. If you deviate, ensure `transformers` ↔ `torch` ↔ `peft` are compatible.

---

## Acknowledgements

- OpenAI Whisper — <https://github.com/openai/whisper>  
- Hugging Face Transformers — <https://github.com/huggingface/transformers>  
- PEFT / LoRA — <https://github.com/huggingface/peft>  
- evaluate — <https://github.com/huggingface/evaluate>  

---

## License

MIT — see `LICENSE` (included).

<!-- Reference links -->
[whisper]: https://github.com/openai/whisper
[transformers]: https://github.com/huggingface/transformers
[peft]: https://github.com/huggingface/peft
[evaluate]: https://github.com/huggingface/evaluate
[pytorch]: https://pytorch.org/
[cuda]: https://developer.nvidia.com/cuda-zone
[python]: https://www.python.org/

[whisper-doc]: https://huggingface.co/docs/transformers/model_doc/whisper

[whisper-badge]: https://img.shields.io/badge/Whisper-OpenAI-412991?logo=openai&logoColor=white
[transformers-badge]: https://img.shields.io/badge/Transformers-HuggingFace-ffcc4d?logo=huggingface&logoColor=black
[peft-badge]: https://img.shields.io/badge/PEFT-LoRA-ffcc4d?logo=huggingface&logoColor=black
[evaluate-badge]: https://img.shields.io/badge/evaluate-Metrics-ffcc4d?logo=huggingface&logoColor=black
[pytorch-badge]: https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white
[cuda-badge]: https://img.shields.io/badge/CUDA-12.x-76b900?logo=nvidia&logoColor=white
[python-badge]: https://img.shields.io/badge/Python-3.10—3.12-3776ab?logo=python&logoColor=white
