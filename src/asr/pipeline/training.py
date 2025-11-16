import pysrt
import json
import random
from pathlib import Path
from typing import List
import torch as T
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          WhisperForConditionalGeneration, WhisperProcessor)
from src.asr.model.trainer import WhisperSafeTrainer



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