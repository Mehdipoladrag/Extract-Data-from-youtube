import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import evaluate
import numpy as np
import pysrt
import soundfile as sf
import torch as T
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

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


class SRTPreprocessor:
    """SRT utilities: (1) convert SRT time to seconds, (2) normalize text (spaces, newlines)."""

    @staticmethod
    def srt_time_to_sec(t) -> float:
        return t.hours * 3600 + t.minutes * 60 + t.seconds + t.milliseconds / 1000.0

    @staticmethod
    def clean_text(s: str) -> str:
        s = s.replace("\u200c", " ").replace("‌", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s


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


class ModelBuilderProcessor:
    """Create Whisper processor/model, set language/task, disable cache (GC), apply LoRA."""

    @staticmethod
    def build_model():
        processor = WhisperProcessor.from_pretrained(MODEL_NAME)
        if hasattr(processor, "feature_extractor"):
            try:
                processor.feature_extractor.return_attention_mask = False
            except Exception:
                pass
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

        model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
        model.generation_config.language = LANG
        model.generation_config.task = TASK
        model.config.use_cache = False

        peft_cfg = LoraConfig(
            task_type="SEQ_2_SEQ_LM",
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()
        return model, processor


class MapExampler:
    """Convert one manifest record to model inputs: speech features + tokenized labels."""

    def __init__(self, processor: WhisperProcessor):
        self.processor = processor

    def mapping(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        a, sr = LoadVoice(rec).load_audio()
        x = self.processor.feature_extractor(a, sampling_rate=sr, return_tensors=None)
        feats = x["input_features"] if isinstance(x, dict) else x.input_features
        if isinstance(feats, np.ndarray) and feats.ndim == 3:
            feats = feats[0]
        feats = feats.tolist()
        y = self.processor.tokenizer(rec["text"], add_special_tokens=True)
        return {"input_features": feats, "labels": y["input_ids"]}


class DataCollatorSpeechSeq2Seq:
    """Batch collator: pad (feature_dim, time) to same length and mask padded labels to -100."""

    def __init__(
        self,
        tokenizer: Any,
        feature_dim: int = 80,
        pad_to_multiple_of: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.feature_dim = feature_dim
        self.pad_to_multiple_of = pad_to_multiple_of
        self._pad_id = (
            tokenizer.pad_token_id
            if getattr(tokenizer, "pad_token_id", None) is not None
            else tokenizer.eos_token_id
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, T.Tensor]:
        feats = [self._prepare_feature(f["input_features"]) for f in features]
        input_features = self._pad_features(feats)
        labels = [self._prepare_labels(f["labels"]) for f in features]
        labels_padded = self._pad_labels(labels)
        return {"input_features": input_features, "labels": labels_padded}

    def _prepare_feature(self, x: Any) -> T.Tensor:
        if isinstance(x, list):
            x = np.asarray(x, dtype=np.float32)
        if isinstance(x, np.ndarray):
            x = T.tensor(x, dtype=T.float32)
        if not isinstance(x, T.Tensor):
            x = T.tensor(x, dtype=T.float32)
        if x.ndim == 3 and x.shape[0] == 1:
            x = x.squeeze(0)
        if x.ndim == 1:
            raise ValueError(
                f"input_features is 1D {tuple(x.shape)}; expected (feature_dim, time)."
            )
        if x.ndim != 2:
            raise ValueError(
                f"Unexpected input_features ndim={x.ndim}, shape={tuple(x.shape)}"
            )
        if x.shape[0] != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, got {x.shape[0]} with shape {tuple(x.shape)}"
            )
        return x

    def _pad_features(self, feats: List[T.Tensor]) -> T.Tensor:
        times = [x.shape[1] for x in feats]
        max_time = max(times)
        if self.pad_to_multiple_of:
            remainder = max_time % self.pad_to_multiple_of
            if remainder:
                max_time += self.pad_to_multiple_of - remainder
        padded = []
        for x in feats:
            pad_len = max_time - x.shape[1]
            if pad_len > 0:
                pad = T.zeros((x.shape[0], pad_len), dtype=x.dtype, device=x.device)
                x = T.cat([x, pad], dim=1)
            padded.append(x)
        return T.stack(padded, dim=0)

    def _prepare_labels(self, ids: Any) -> T.Tensor:
        if isinstance(ids, list):
            return T.tensor(ids, dtype=T.long)
        if isinstance(ids, np.ndarray):
            return T.from_numpy(ids.astype(np.int64))
        if isinstance(ids, T.Tensor):
            return ids.to(dtype=T.long)
        return T.tensor(list(ids), dtype=T.long)

    def _pad_labels(self, labels: List[T.Tensor]) -> T.Tensor:
        padded = T.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self._pad_id
        )
        padded = padded.masked_fill(padded == self._pad_id, -100)
        return padded


class ComputeMetrics:
    """Compute CER during eval by decoding predictions/labels and normalizing pad tokens."""

    def __init__(self, processor):
        self.processor = processor
        self.cer_metric = evaluate.load("cer")

    def __call__(self, pred):
        pred_ids = np.asarray(pred.predictions).astype(np.int64)
        label_ids = np.asarray(pred.label_ids).astype(np.int64)
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        pred_str = self.processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        label_str = self.processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )
        cer_score = self.cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer_score}


class WhisperSafeTrainer(Seq2SeqTrainer):
    """Custom Trainer: use base model under PEFT; feed input_features directly; stable generate."""

    def _peft_base(self, model):
        return getattr(model, "base_model", model)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        base = self._peft_base(model)
        input_features = inputs.pop("input_features")
        labels = inputs.get("labels")
        if self.args.gradient_checkpointing and isinstance(input_features, T.Tensor):
            input_features = input_features.requires_grad_(True)
        outputs = base(input_features=input_features, labels=labels)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        return (loss, outputs) if return_outputs else loss

    @T.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        base = self._peft_base(model)
        has_labels = "labels" in inputs
        input_features = inputs["input_features"]
        labels = inputs["labels"] if has_labels else None

        gen_kwargs = dict(
            language=LANG,
            task=TASK,
            do_sample=False,
            num_beams=1,
            no_repeat_ngram_size=4,
            repetition_penalty=1.1,
            max_new_tokens=self.args.generation_max_length or 225,
        )

        if self.args.predict_with_generate and not prediction_loss_only:
            generated_tokens = base.generate(
                input_features=input_features, **gen_kwargs
            )
        else:
            generated_tokens = None

        with self.compute_loss_context_manager():
            outputs = (
                base(input_features=input_features, labels=labels)
                if has_labels
                else base(input_features=input_features)
            )
            loss = (
                (
                    outputs["loss"] if isinstance(outputs, dict) else outputs.loss
                ).detach()
                if has_labels
                else None
            )

        if generated_tokens is not None and generated_tokens.ndim == 1:
            generated_tokens = generated_tokens.unsqueeze(0)

        return (loss, generated_tokens, labels if has_labels else None)


class WhisperTrainingPipeline:
    """End-to-end pipeline: setup, dataset prep/map, model/trainer build, train & save."""

    def __init__(
        self,
        manifest_path: Path,
        build_manifest_fn,
        build_model_and_processor,
        map_example_fn,
        data_collator_cls,
        compute_metrics_fn_factory,
        output_dir: str = "runs/whisper-fa-lora",
        train_batch_size: int = 16,
        eval_batch_size: int = 8,
        learning_rate: float = 1e-4,
        warmup_steps: int = 50,
        max_steps: int = 800,
        grad_accum_steps: int = 1,
        fp16: bool = True,
        eval_steps: int = 100,
        save_steps: int = 100,
        logging_steps: int = 25,
        gen_max_len: int = 225,
        gen_num_beams: int = 1,
        save_total_limit: int = 2,
        seed: int = 42,
    ):
        self.manifest_path = manifest_path
        self.build_manifest_fn = build_manifest_fn
        self.build_model_and_processor = build_model_and_processor
        self.map_example_fn = map_example_fn
        self.data_collator_cls = data_collator_cls
        self.compute_metrics_fn_factory = compute_metrics_fn_factory

        self.output_dir = output_dir
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.grad_accum_steps = grad_accum_steps
        self.fp16 = fp16
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.gen_max_len = gen_max_len
        self.gen_num_beams = gen_num_beams
        self.save_total_limit = save_total_limit
        self.seed = seed

        self.model = None
        self.processor = None
        self.trainer = None
        self.train_ds = None
        self.valid_ds = None

    def run(self):
        self._setup_torch()
        self._prepare_datasets()
        self._build_model()
        self._map_datasets()
        self._build_trainer()
        first = next(iter(self.trainer.get_train_dataloader()))
        print("BATCH KEYS:", list(first.keys()))
        self._train_and_save()

    def _setup_torch(self):
        try:
            T.set_float32_matmul_precision("high")
        except Exception:
            pass
        use_cuda = T.cuda.is_available()
        use_bf16 = use_cuda and getattr(T.cuda, "is_bf16_supported", lambda: False)()
        print(f"torch: {T.__version__} | cuda: {use_cuda} | bf16: {use_bf16}")
        if use_cuda:
            try:
                print("device:", T.cuda.get_device_name(0))
            except Exception:
                pass
        random.seed(self.seed)

    def _prepare_datasets(self):
        self.build_manifest_fn()
        recs: List[dict] = [
            json.loads(l) for l in self.manifest_path.open(encoding="utf-8")
        ]
        if not recs:
            raise SystemExit("manifest.jsonl is empty.")
        random.shuffle(recs)
        split = int(0.8 * len(recs)) if len(recs) > 1 else len(recs)
        train_recs = recs[:split]
        valid_recs = recs[split:] if split < len(recs) else recs
        self.train_ds = Dataset.from_list(train_recs)
        self.valid_ds = Dataset.from_list(valid_recs)

    def _build_model(self):
        self.model, self.processor = self.build_model_and_processor()
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        self.model.generation_config.max_length = self.gen_max_len
        self.model.generation_config.num_beams = self.gen_num_beams

    def _map_datasets(self):
        self.train_ds = self.train_ds.map(
            lambda rec: self.map_example_fn(rec, self.processor),
            remove_columns=self.train_ds.column_names,
        )
        self.valid_ds = self.valid_ds.map(
            lambda rec: self.map_example_fn(rec, self.processor),
            remove_columns=self.valid_ds.column_names,
        )
        keep_cols = ["input_features", "labels"]
        self.train_ds = self.train_ds.remove_columns(
            [c for c in self.train_ds.column_names if c not in keep_cols]
        )
        self.valid_ds = self.valid_ds.remove_columns(
            [c for c in self.valid_ds.column_names if c not in keep_cols]
        )
        self.train_ds = self.train_ds.with_format(type="torch", columns=keep_cols)
        self.valid_ds = self.valid_ds.with_format(type="torch", columns=keep_cols)

    def _build_trainer(self):
        collator = self.data_collator_cls(tokenizer=self.processor.tokenizer)
        args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            gradient_accumulation_steps=self.grad_accum_steps,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            fp16=self.fp16,
            eval_steps=self.eval_steps,
            save_steps=self.save_steps,
            logging_steps=self.logging_steps,
            predict_with_generate=True,
            generation_max_length=self.gen_max_len,
            generation_num_beams=self.gen_num_beams,
            save_total_limit=self.save_total_limit,
            load_best_model_at_end=False,
            report_to=["tensorboard"],
            remove_unused_columns=False,
            label_names=["labels"],
        )
        args.generation_config = self.model.generation_config

        self.trainer = WhisperSafeTrainer(
            model=self.model,
            args=args,
            train_dataset=self.train_ds,
            eval_dataset=self.valid_ds,
            data_collator=collator,
            compute_metrics=self.compute_metrics_fn_factory(self.processor),
        )

    def _train_and_save(self):
        self.trainer.train()
        out_dir = Path(self.output_dir) / "final"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.trainer.save_model(str(out_dir))
        self.processor.save_pretrained(str(out_dir))
        print("\n✅ Done. Saved to:", out_dir.resolve())


class WhisperFineTuningApp:
    """Thin app wrapper to assemble and run the training pipeline."""

    def __init__(self):
        self.manifest_path = MANIFEST
        self.output_dir = "runs/whisper-fa-lora"

    def build_manifest(self):
        mb = ManifestBuilder()
        mb.build_if_needed()

    def build_model_and_processor(self):
        return ModelBuilderProcessor.build_model()

    def map_example(self, rec, processor):
        return MapExampler(processor).mapping(rec)

    def run(self):
        pipeline = WhisperTrainingPipeline(
            manifest_path=self.manifest_path,
            build_manifest_fn=self.build_manifest,
            build_model_and_processor=self.build_model_and_processor,
            map_example_fn=self.map_example,
            data_collator_cls=DataCollatorSpeechSeq2Seq,
            compute_metrics_fn_factory=ComputeMetrics,
            output_dir=self.output_dir,
        )
        pipeline.run()


if __name__ == "__main__":
    app = WhisperFineTuningApp()
    app.run()
