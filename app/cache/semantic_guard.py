from pathlib import Path
import re
from typing import Tuple
import yaml

from app.cache.policies import INFORMATIONAL_PREFIXES, is_informational_query
from app.core.config import settings

NEGATION_CANONICAL_MAP = {
    "not": "not",
    "no": "no",
    "never": "never",
    "don't": "don't",
    "dont": "don't",
    "doesn't": "doesn't",
    "doesnt": "doesn't",
    "can't": "can't",
    "cant": "can't",
    "without": "without",
    "cancel": "cancel",
    "stop": "stop",
}

NEGATION_REGEX = re.compile(
    r"\b(not|no|never|don't|dont|doesn't|doesnt|can't|cant|without|cancel|stop)\b",
    re.IGNORECASE,
)

NUMBER_REGEX = re.compile(r"\b\d+(?:\.\d+)?\b")

# Fallback default antonym pairs if config file is absent
DEFAULT_ANTONYM_PAIRS: list[tuple[str, str]] = [
    ("enable", "disable"),
    ("buy", "sell"),
    ("increase", "decrease"),
    ("lock", "unlock"),
    ("open", "close"),
    ("allow", "deny"),
    ("add", "remove"),
    ("on", "off"),
    ("start", "pause"),
    ("approve", "reject"),
    ("upload", "download"),
    ("safe", "dangerous"),
    ("accept", "decline"),
    ("encrypt", "decrypt"),
    ("subscribe", "unsubscribe"),
    ("before", "after"),
    ("win", "lose"),
    ("include", "exclude"),
]

IRREGULAR_VARIANTS: dict[str, set[str]] = {
    "buy": {"bought", "buys", "buying", "buyer", "buyers"},
    "sell": {"sold", "sells", "selling", "seller", "sellers"},
    "win": {"won", "wins", "winning", "winner", "winners"},
    "lose": {"lost", "loses", "losing", "loser", "losers"},
    "safe": {"safer", "safest", "safely", "safety"},
    "dangerous": {"danger", "dangers", "dangerously"},
    "deny": {"denies", "denied", "denying"},
    "encrypt": {"encryption"},
    "decrypt": {"decryption"},
    "subscribe": {"subscription", "subscriptions"},
    "unsubscribe": {"unsubscription", "unsubscriptions"},
    "approve": {"approval", "approvals"},
    "reject": {"rejection", "rejections"},
    "accept": {"acceptance"},
    "include": {"inclusion"},
    "exclude": {"exclusion"},
}


def get_word_variants(word: str) -> set[str]:
    """Generates standard inflected forms and irregular variants for an antonym keyword."""
    w = word.lower().strip()
    variants = {w}
    if w in IRREGULAR_VARIANTS:
        variants.update(IRREGULAR_VARIANTS[w])
    if w in ("on", "off", "before", "after") or len(w) < 3:
        return variants

    variants.add(w + "s")
    if w.endswith("e"):
        variants.add(w + "d")
        variants.add(w[:-1] + "ing")
    elif w.endswith("y"):
        variants.add(w[:-1] + "ies")
        variants.add(w[:-1] + "ied")
        variants.add(w + "ing")
    else:
        variants.add(w + "ed")
        variants.add(w + "ing")

    return variants


def load_antonym_pairs(config_path: str | Path | None = None) -> list[tuple[str, str]]:
    """Loads antonym pairs from YAML configuration file, falling back to built-in list."""
    path = config_path or getattr(settings, "ANTONYMS_CONFIG_PATH", "config/antonyms.yaml")
    p = Path(path)
    if not p.is_absolute():
        candidates = [
            p,
            Path(__file__).resolve().parent.parent.parent / p,
            Path(__file__).resolve().parent.parent / p,
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                p = c
                break

    if p.exists() and p.is_file():
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            raw_pairs = (
                data.get("antonym_pairs")
                or data.get("antonyms")
                or (data if isinstance(data, list) else [])
            )
            parsed: list[tuple[str, str]] = []
            for item in raw_pairs:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    def _to_word(val: object) -> str:
                        if isinstance(val, bool):
                            return "on" if val else "off"
                        return str(val).strip().lower()

                    w1 = _to_word(item[0])
                    w2 = _to_word(item[1])
                    if w1 and w2:
                        parsed.append((w1, w2))
            if parsed:
                return parsed
        except Exception:
            pass

    return list(DEFAULT_ANTONYM_PAIRS)


_ANTONYM_PAIRS_COMPILED: list[tuple[str, str, set[str], set[str]]] | None = None


def get_compiled_antonym_pairs() -> list[tuple[str, str, set[str], set[str]]]:
    """Returns cached list of antonym pairs with precompiled variant sets."""
    global _ANTONYM_PAIRS_COMPILED
    if _ANTONYM_PAIRS_COMPILED is None:
        pairs = load_antonym_pairs()
        compiled = []
        for w1, w2 in pairs:
            compiled.append((w1, w2, get_word_variants(w1), get_word_variants(w2)))
        _ANTONYM_PAIRS_COMPILED = compiled
    return _ANTONYM_PAIRS_COMPILED


def reset_antonym_cache() -> None:
    """Resets cached compiled antonyms for dynamic configuration updates in tests."""
    global _ANTONYM_PAIRS_COMPILED
    _ANTONYM_PAIRS_COMPILED = None


def extract_negation_words(text: str) -> set[str]:
    """Extracts normalized negation words from text."""
    clean_text = text.lower().replace("’", "'")
    matches = NEGATION_REGEX.findall(clean_text)
    return {NEGATION_CANONICAL_MAP.get(m.lower(), m.lower()) for m in matches}


def extract_numbers(text: str) -> list[str]:
    """Extracts sequence of normalized number tokens from text."""
    matches = NUMBER_REGEX.findall(text)
    normalized: list[str] = []
    for m in matches:
        try:
            val = float(m)
            if val.is_integer():
                normalized.append(str(int(val)))
            else:
                normalized.append(str(val))
        except ValueError:
            normalized.append(m)
    return sorted(normalized)


def check_antonym_guard(query_text: str, cached_text: str) -> Tuple[bool, str | None]:
    """Verifies that query_text and cached_text do not contain conflicting members of any antonym pair.
    
    Returns (True, None) if check passes.
    Returns (False, "guard") if rejected due to conflicting antonyms.
    """
    q_tokens = set(t.strip("'") for t in re.findall(r"\b[a-zA-Z0-9']+\b", query_text.lower()))
    c_tokens = set(t.strip("'") for t in re.findall(r"\b[a-zA-Z0-9']+\b", cached_text.lower()))

    compiled_pairs = get_compiled_antonym_pairs()
    for w1, w2, v1, v2 in compiled_pairs:
        q_has_w1 = bool(q_tokens & v1)
        q_has_w2 = bool(q_tokens & v2)
        c_has_w1 = bool(c_tokens & v1)
        c_has_w2 = bool(c_tokens & v2)

        q_members = ({w1} if q_has_w1 else set()) | ({w2} if q_has_w2 else set())
        c_members = ({w1} if c_has_w1 else set()) | ({w2} if c_has_w2 else set())

        # If both texts mention at least one member of this pair and they differ:
        if q_members and c_members and q_members != c_members:
            return False, "guard"

    return True, None


def check_semantic_guard(query_text: str, cached_text: str) -> Tuple[bool, str | None]:
    """Verifies that query_text and cached_text share identical negation words, numbers, and antonym pairs.
    
    Returns (True, None) if guard passes.
    Returns (False, "guard") if rejected due to negation, number, or antonym discrepancies.
    """
    query_negations = extract_negation_words(query_text)
    cached_negations = extract_negation_words(cached_text)

    if query_negations != cached_negations:
        return False, "guard"

    query_numbers = extract_numbers(query_text)
    cached_numbers = extract_numbers(cached_text)

    if query_numbers != cached_numbers:
        return False, "guard"

    antonym_ok, antonym_reason = check_antonym_guard(query_text, cached_text)
    if not antonym_ok:
        return False, antonym_reason

    return True, None
