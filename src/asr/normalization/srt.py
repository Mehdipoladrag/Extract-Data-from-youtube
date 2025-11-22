import re
from .fa import FaNormalization

FA_NORM_SOFT = FaNormalization(strip_punct=False)


class SRTPreprocessor:
    """SRT utilities: (1) convert SRT time to seconds, (2) normalize text (spaces, newlines)."""

    @staticmethod
    def srt_time_to_sec(t) -> float:
        return t.hours * 3600 + t.minutes * 60 + t.seconds + t.milliseconds / 1000.0

    @staticmethod
    def clean_text(s: str) -> str:
        if not s:
            return ""
        s = s.replace("\n", " ")
        return FA_NORM_SOFT.normalize(s)
