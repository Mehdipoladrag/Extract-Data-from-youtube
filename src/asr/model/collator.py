from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pysrt
import torch as T
from asr.configs.main_conf import LANG, TASK, MODEL_NAME, MANIFEST, AUDIO_DIR, SUB_DIR, STR_SUFFIX





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