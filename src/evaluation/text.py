"""Conservative text scoring. Cross-script equivalences are never invented."""
from __future__ import annotations

import unicodedata

from rapidfuzz.distance import Levenshtein

NORMALIZATION_VERSION = "hi-en-nfc-v1"
PIER_SOURCE = "enesyugan/PIER-CodeSwitching-Evaluation@09535fa0311cfec98176d1a4f7bec82094bb0d4f"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    # Apostrophe and hyphen glyph variants are orthographic, not transliterations.
    text = text.translate(str.maketrans({"’": "'", "‘": "'", "‐": "-", "‑": "-"}))
    chars = []
    for i, c in enumerate(text):
        left = text[i - 1] if i else ""
        right = text[i + 1] if i + 1 < len(text) else ""
        internal = c in "'-" and left.isalnum() and right.isalnum()
        numeric = c in ".,:/" and left.isdecimal() and right.isdecimal()
        if unicodedata.category(c).startswith("P") and not (internal or numeric):
            c = " "
        elif "LATIN" in unicodedata.name(c, ""):
            c = c.casefold()
        chars.append(c)
    return " ".join("".join(chars).split())


def edit_score(reference: str | None, hypothesis: str | None):
    if reference is None or hypothesis is None:
        return {"status": "missing_input", "wer": None, "cer": None}
    ref, hyp = normalize(reference), normalize(hypothesis)
    rt, ht = ref.split(), hyp.split()
    if not rt:
        return {"status": "invalid_empty_reference", "wer": None, "cer": None}
    operations = Levenshtein.editops(rt, ht)
    counts = {name: sum(op.tag == tag for op in operations) for name, tag in
              (("substitutions", "replace"), ("deletions", "delete"), ("insertions", "insert"))}
    rc, hc = "".join(rt), "".join(ht)
    return {"status": "computed_unvalidated", "normalization_version": NORMALIZATION_VERSION,
            "reference_normalized": ref, "hypothesis_normalized": hyp,
            "reference_tokens": rt, "hypothesis_tokens": ht,
            "reference_words": len(rt), "reference_characters": len(rc), **counts,
            "character_errors": Levenshtein.distance(rc, hc),
            "wer": len(operations) / len(rt), "cer": Levenshtein.distance(rc, hc) / len(rc),
            "alignment": [{"operation": op.tag, "reference_index": op.src_pos,
                           "hypothesis_index": op.dest_pos} for op in operations]}


def pier(reference_tokens, hypothesis_tokens, poi_indices):
    """Official get_idsh semantics, with explicit N/A instead of zero for no POI.

    Align the COMPLETE token sequences. Insertion src_pos is the reference gap
    index. Trailing insertions count iff the final reference token is a POI.
    RapidFuzz is pinned in execution provenance because edit ties affect PIER.
    """
    n = len(reference_tokens)
    indices = set(poi_indices)
    if any(not isinstance(i, int) or not 0 <= i < n for i in indices):
        raise ValueError("Invalid POI index")
    if not indices:
        return {"status": "not_applicable", "pier": None, "poi_words": 0}
    ops = [op for op in Levenshtein.editops(reference_tokens, hypothesis_tokens)
           if op.src_pos in indices or (n - 1 in indices and op.src_pos >= n)]
    return {"status": "computed_unvalidated", "pier": len(ops) / len(indices),
            "poi_words": len(indices), "substitutions": sum(o.tag == "replace" for o in ops),
            "deletions": sum(o.tag == "delete" for o in ops),
            "insertions": sum(o.tag == "insert" for o in ops), "implementation_reference": PIER_SOURCE}


def checked_language(annotation, raw_text):
    from .common import digest
    if not annotation or annotation.get("verified") is not True:
        return None
    tokens = normalize(raw_text).split()
    if annotation.get("text_sha256") != digest(raw_text.encode("utf-8")):
        raise ValueError("Language annotation belongs to different text")
    labels = annotation.get("labels", [])
    if annotation.get("tokens") != tokens or len(labels) != len(tokens):
        raise ValueError("Language annotation/token alignment mismatch")
    if any(t not in ("HI", "EN", "OTHER", "MIXED", "UNCERTAIN") for t in labels):
        raise ValueError("Invalid language label")
    if "MIXED" in labels or "UNCERTAIN" in labels:
        return None
    return labels
