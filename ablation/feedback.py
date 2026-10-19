"""Change only the reward label shown in prompts, using dowhen instrumentation."""

from contextlib import contextmanager

from dowhen import do
from num2words import num2words

LANGUAGES = ("numeric", "en", "fr", "zh")
_ZH_DIGITS = "零一二三四五六七八九"
_ZH_SMALL_UNITS = ("", "十", "百", "千")
_ZH_BIG_UNITS = ("", "万", "亿", "兆")


def _zh_group(number: int) -> str:
    result = ""
    pending_zero = False
    for power in (3, 2, 1, 0):
        digit = number // (10**power) % 10
        if digit:
            if pending_zero and result:
                result += "零"
            if not (power == 1 and digit == 1 and not result):
                result += _ZH_DIGITS[digit]
            result += _ZH_SMALL_UNITS[power]
            pending_zero = False
        elif result:
            pending_zero = True
    return result


def _zh_integer(number: int) -> str:
    if number == 0:
        return "零"
    if number < 0:
        return "负" + _zh_integer(-number)
    groups = []
    while number:
        groups.append(number % 10000)
        number //= 10000
    if len(groups) > len(_ZH_BIG_UNITS):
        raise ValueError("Chinese reward renderer supports magnitudes below 10**16")
    result = ""
    pending_zero = False
    for index in range(len(groups) - 1, -1, -1):
        group = groups[index]
        if not group:
            pending_zero = bool(result)
            continue
        if result and (pending_zero or group < 1000):
            result += "零"
        result += _zh_group(group) + _ZH_BIG_UNITS[index]
        pending_zero = False
    return result


def spell_reward(score: int, language: str) -> str:
    """Spell an integer reward without changing its stored numeric value."""
    if isinstance(score, bool) or int(score) != score:
        raise ValueError(f"Expected an integer-valued reward, got {score!r}")
    value = int(score)
    if language == "numeric":
        return str(value)
    if language == "zh":
        return _zh_integer(value)
    if language in ("en", "fr"):
        return num2words(value, lang=language)
    raise ValueError(f"Unknown feedback language: {language}")


def format_feedback(score: int) -> str:
    """The single numeric feedback boundary instrumented by dowhen."""
    label = str(score)
    return label


@contextmanager
def feedback_language(language: str):
    """Temporarily replace the local `label` just before it is returned."""
    if language not in LANGUAGES:
        raise ValueError(f"Unknown feedback language: {language}")
    if language == "numeric":
        yield
        return

    def replace_label(score):
        return {"label": spell_reward(score, language)}

    handler = do(replace_label).when(format_feedback, "return label")
    try:
        yield
    finally:
        handler.remove()
