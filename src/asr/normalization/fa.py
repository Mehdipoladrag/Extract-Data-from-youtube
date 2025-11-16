# -*- coding: utf-8 -*-
from __future__ import annotations
import re
from typing import Final, Optional

class FaNormalization:
    

    _AR2FA: Final = str.maketrans({
        "ي": "ی", "ى": "ی", "ئ": "ی", "ك": "ک",
        "۰":"0","۱":"1","۲":"2","۳":"3","۴":"4","۵":"5","۶":"6","۷":"7","۸":"8","۹":"9",
    })

    _DIACRITICS_RANGE: Final = "".join(chr(c) for c in range(0x064B, 0x065F)) + "\u0640"
    _PUNCT: Final = r"""!"#$%&'()*+,\-./:;<=>?@[\]^_`{|}~،؛«»؟…"""
    _CONTROL_CHARS: Final = "\u200e\u200f\u202a\u202b\u202c\u202d\u202e"

    _re_space = re.compile(r"\s+")
    _re_diacritics = re.compile(f"[{_DIACRITICS_RANGE}]")
    _re_punct = re.compile(f"[{re.escape(_PUNCT)}]")
    _re_controls = re.compile(f"[{re.escape(_CONTROL_CHARS)}]")

    def __init__(
        self,
        *,
        map_arabic_chars: bool = True,
        remove_diacritics: bool = True,
        unify_zwjn: bool = True,
        strip_punct: bool = True,
        collapse_spaces: bool = True,
        strip_controls: bool = True,
    ) -> None:
        
        self.map_arabic_chars = map_arabic_chars
        self.remove_diacritics = remove_diacritics
        self.unify_zwjn = unify_zwjn
        self.strip_punct = strip_punct
        self.collapse_spaces = collapse_spaces
        self.strip_controls = strip_controls

    def normalize(
        self,
        s: Optional[str],
        *,
        map_arabic_chars: Optional[bool] = None,
        remove_diacritics: Optional[bool] = None,
        unify_zwjn: Optional[bool] = None,
        strip_punct: Optional[bool] = None,
        collapse_spaces: Optional[bool] = None,
        strip_controls: Optional[bool] = None,
    ) -> str:
 
        if not s:
            return ""
        out = s.strip()

        map_arabic_chars = self.map_arabic_chars if map_arabic_chars is None else map_arabic_chars
        remove_diacritics = self.remove_diacritics if remove_diacritics is None else remove_diacritics
        unify_zwjn = self.unify_zwjn if unify_zwjn is None else unify_zwjn
        strip_punct = self.strip_punct if strip_punct is None else strip_punct
        collapse_spaces = self.collapse_spaces if collapse_spaces is None else collapse_spaces
        strip_controls = self.strip_controls if strip_controls is None else strip_controls

        if strip_controls:
            out = self._re_controls.sub("", out)

        if map_arabic_chars:
            out = out.translate(self._AR2FA)

        if remove_diacritics:
            out = self._re_diacritics.sub("", out)

        if unify_zwjn:
            out = out.replace("\u200c", " ").replace("‌", " ")

        if strip_punct:
            out = self._re_punct.sub(" ", out)

        if collapse_spaces:
            out = self._re_space.sub(" ", out).strip()

        return out
