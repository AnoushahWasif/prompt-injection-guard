"""Undo the text tricks used to hide an injection from a filter, before anything scores it.

A classifier only sees characters. "ig​nore previous instructions" (zero-width space),
"ｉｇｎｏｒｅ" (fullwidth) and "іgnore" (Cyrillic і) all read as "ignore" to a person and to most
LLMs, but not to a model trained on plain text. The same normalised text goes to training and
to the live guard, so the two never disagree.

What was undone is returned too: hidden characters in a prompt are a signal on their own.
"""

import re
import unicodedata
from dataclasses import dataclass

MAX_CHARS = 20_000

# Unicode "tag" characters (U+E0020-E007E) mirror printable ASCII but render as nothing.
# They've been used to smuggle instructions past people reading a prompt ("ASCII smuggling"),
# so they're decoded into visible text rather than dropped.
TAG_FIRST, TAG_LAST = 0xE0020, 0xE007E
TAG_OFFSET = 0xE0000

# Variation selectors are combining marks (category Mn), not format characters, so the Cf check
# below misses them. They can carry hidden data too.
VARIATION_SELECTORS = re.compile("[︀-️\U000e0100-\U000e01ef]")

# Lookalikes for Latin letters. Only applied inside words that also contain Latin letters, so
# real Russian or Greek text is left alone.
CONFUSABLES = str.maketrans(
    {
        # Cyrillic
        "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
        "і": "i", "ј": "j", "ѕ": "s", "һ": "h", "ԁ": "d", "ԛ": "q", "ԝ": "w",
        "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
        "Р": "P", "С": "C", "Т": "T", "Х": "X", "І": "I", "Ј": "J", "Ѕ": "S",
        # Greek
        "α": "a", "ο": "o", "ι": "i", "κ": "k", "ν": "v", "ρ": "p", "τ": "t",
        "υ": "u", "χ": "x", "Α": "A", "Β": "B", "Ε": "E", "Η": "H", "Ι": "I",
        "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Χ": "X",
        "Υ": "Y", "Ζ": "Z",
    }
)  # fmt: skip
WORD = re.compile(r"\w+")
LATIN = re.compile(r"[A-Za-z]")
HORIZONTAL_SPACE = re.compile(r"[^\S\n]+")
BLANK_LINES = re.compile(r"\n\s*\n+")


@dataclass(frozen=True)
class Normalized:
    text: str
    invisible_removed: int = 0
    tag_chars_decoded: int = 0
    confusables_replaced: int = 0
    width_or_compat_changed: bool = False

    @property
    def suspicious(self) -> bool:
        """Characters a normal user doesn't type were found. Not proof, but worth a look."""
        return bool(self.invisible_removed or self.tag_chars_decoded or self.confusables_replaced)


def _decode_tags(text: str) -> tuple[str, int]:
    count = 0
    out = []
    for ch in text:
        code = ord(ch)
        if TAG_FIRST <= code <= TAG_LAST:
            out.append(chr(code - TAG_OFFSET))
            count += 1
        else:
            out.append(ch)
    return "".join(out), count


def _fix_mixed_script(text: str) -> tuple[str, int]:
    count = 0

    def fix(match: re.Match) -> str:
        nonlocal count
        word = match.group()
        if not LATIN.search(word):
            return word
        fixed = word.translate(CONFUSABLES)
        count += sum(a != b for a, b in zip(word, fixed, strict=True))
        return fixed

    return WORD.sub(fix, text), count


def normalize(text: str) -> Normalized:
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")
    # Checked before any work, so a huge input can't make normalising itself slow.
    if len(text) > MAX_CHARS:
        raise ValueError(f"text longer than {MAX_CHARS} characters")

    text, tags = _decode_tags(text)

    # Tag characters are format characters (Cf) too, so they had to be decoded first.
    before = len(text)
    text = VARIATION_SELECTORS.sub("", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    invisible = before - len(text)

    # NFKC folds fullwidth letters, ligatures and similar compatibility forms into plain ones.
    folded = unicodedata.normalize("NFKC", text)
    changed = folded != text

    text, confusables = _fix_mixed_script(folded)
    text = HORIZONTAL_SPACE.sub(" ", text)
    text = BLANK_LINES.sub("\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n")).strip()
    return Normalized(text, invisible, tags, confusables, changed)
