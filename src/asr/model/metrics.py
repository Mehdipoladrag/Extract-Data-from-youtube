import evaluate
import numpy as np
import pysrt
from datasets import Dataset
from src.asr.normalization.fa import FaNormalization
from asr.configs.main_conf import (
    LANG,
    TASK,
    MODEL_NAME,
    MANIFEST,
    AUDIO_DIR,
    SUB_DIR,
    STR_SUFFIX,
)


class ComputeMetrics:
    """Compute CER & WER with Persian normalization (pad tokens handled)."""

    def __init__(self, processor):
        self.processor = processor
        self.cer_metric = evaluate.load("cer")
        self.wer_metric = evaluate.load("wer")
        self._norm = FaNormalization()

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

        pred_norm = [self._norm.for_metric(s) for s in pred_str]
        label_norm = [self._norm.for_metric(s) for s in label_str]

        cer_score = self.cer_metric.compute(
            predictions=pred_norm, references=label_norm
        )
        wer_score = self.wer_metric.compute(
            predictions=pred_norm, references=label_norm
        )

        return {"cer": cer_score, "wer": wer_score}

    def _compute_metrics_pair(self, hyp: str, ref: str):
        hyp_norm = self._norm.for_metric(hyp)
        ref_norm = self._norm.for_metric(ref)

        cer = self._cer_metric.compute(predictions=[hyp_norm], references=[ref_norm])
        wer = self._wer_metric.compute(predictions=[hyp_norm], references=[ref_norm])
        return cer, wer, hyp_norm, ref_norm
