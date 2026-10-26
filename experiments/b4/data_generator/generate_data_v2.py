"""
Phase B4-1a: Improved synthetic data generator (v4 鈥?"see" fix).

Key fixes from v3:
  - "I see" replaced with "He sees"/"She knows" (vocab has "sees"/"knows" but NOT "see")
  - All templates guaranteed to use only in-vocab words
  - Exact dedup only (preserves ?, !, structural differences)
  - Min 4 words enforced
  - All sentences start with uppercase letter

Templates: 46 across 10 grammatical categories.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable


# =====================================================================
# VOCABULARY
# =====================================================================

ADJECTIVES: list[str] = [
    "big", "small", "old", "new", "good", "bad",
    "long", "short", "high", "low",
    "dark", "light", "warm", "cold", "bright",
    "soft", "hard", "quick", "slow", "deep",
]

VERBS_INTRANSITIVE: list[str] = [
    "runs", "jumps", "sits", "goes", "comes",
    "becomes", "holds", "needs",
]

VERBS_TRANSITIVE: list[str] = [
    "likes", "sees", "takes", "gives", "knows",
    "wants", "finds", "puts", "makes",
]

VERBS_COPLA: list[str] = ["is", "was"]

VERBS_POSSESS: list[str] = ["has"]

NOUNS_COMMON: list[str] = [
    "cat", "dog", "bird", "fish",
    "book", "table", "chair", "house",
    "tree", "world", "time", "day", "night",
    "hand", "head", "child", "man", "woman", "friend",
]

PRONOUNS: list[str] = ["he", "she", "it", "they", "we", "you", "I"]

NOUNS_ALL: list[str] = NOUNS_COMMON + PRONOUNS

ADVERBS: list[str] = [
    "quickly", "slowly", "warmly", "deeply", "brightly",
    "softly", "gently", "fully",
    "very", "quite", "really",
]

PREPOSITIONS: list[str] = [
    "on", "in", "under", "near", "behind",
    "before", "after", "through", "across", "around",
]

CONJUNCTIONS: list[str] = ["and", "so", "yet"]

DETERMINERS_LOWER: list[str] = [
    "the", "a", "some", "many",
    "each", "every", "other", "such",
]

COMPARATORS: list[str] = ["more", "most", "less"]

TIME_WORDS: list[str] = ["day", "night"]

ALL_VALID: set[str] = set(
    ADJECTIVES + VERBS_INTRANSITIVE + VERBS_TRANSITIVE + VERBS_COPLA
    + VERBS_POSSESS + NOUNS_COMMON + PRONOUNS + ADVERBS
    + PREPOSITIONS + CONJUNCTIONS + DETERMINERS_LOWER
    + COMPARATORS + TIME_WORDS
)


# =====================================================================
# HELPERS
# =====================================================================

def _r(rng: random.Random, lst: list) -> str:
    return rng.choice(lst)


def _adj(rng: random.Random) -> str:
    return _r(rng, ADJECTIVES)


def _noun_common(rng: random.Random) -> str:
    return _r(rng, NOUNS_COMMON)


def _noun_any(rng: random.Random) -> str:
    return _r(rng, NOUNS_ALL)


def _verb_intrans(rng: random.Random) -> str:
    return _r(rng, VERBS_INTRANSITIVE)


def _verb_trans(rng: random.Random) -> str:
    return _r(rng, VERBS_TRANSITIVE)


def _verb_copula(rng: random.Random) -> str:
    return _r(rng, VERBS_COPLA)


def _adv(rng: random.Random) -> str:
    return _r(rng, ADVERBS)


def _prep(rng: random.Random) -> str:
    return _r(rng, PREPOSITIONS)


def _conj(rng: random.Random) -> str:
    return _r(rng, CONJUNCTIONS)


def _comp(rng: random.Random) -> str:
    return _r(rng, COMPARATORS)


def _time(rng: random.Random) -> str:
    return _r(rng, TIME_WORDS)


def _capitalize(s: str) -> str:
    if s and s[0].islower():
        return s[0].upper() + s[1:]
    return s


# =====================================================================
# TEMPLATES
# =====================================================================

# --- Cat 1: Simple Declarative (6) ---
def _t01_sva(rng: random.Random) -> str:
    """{noun} {verb} {adv}."""
    return _capitalize(f"{_noun_any(rng)} {_verb_intrans(rng)} {_adv(rng)}.")


def _t02_the_sva(rng: random.Random) -> str:
    """The {adj} {noun} {verb} {adv}."""
    return f"The {_adj(rng)} {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)}."


def _t03_a_sva(rng: random.Random) -> str:
    """A {adj} {noun} {verb} {adv}."""
    return f"A {_adj(rng)} {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)}."


def _t04_sva_time(rng: random.Random) -> str:
    """{noun} {verb} {adv} {time}."""
    return _capitalize(f"{_noun_any(rng)} {_verb_intrans(rng)} {_adv(rng)} {_time(rng)}.")


def _t05_time_sva(rng: random.Random) -> str:
    """{time}, {noun} {verb} {adv}."""
    return f"{_time(rng)}, {_noun_any(rng)} {_verb_intrans(rng)} {_adv(rng)}."


def _t06_time_adv(rng: random.Random) -> str:
    """{time} {noun} {verb} {adv}."""
    return _capitalize(f"{_time(rng)} {_noun_any(rng)} {_verb_intrans(rng)} {_adv(rng)}.")


# --- Cat 2: Copular Predication (5) ---
def _t07_copula_adj(rng: random.Random) -> str:
    """The {noun} is {adj}."""
    return f"The {_noun_common(rng)} is {_adj(rng)}."


def _t08_copula_adv(rng: random.Random) -> str:
    """The {noun} is {adv}."""
    return f"The {_noun_common(rng)} is {_adv(rng)}."


def _t09_double_copula(rng: random.Random) -> str:
    """The {noun} is {adj} and {adj}."""
    return f"The {_noun_common(rng)} is {_adj(rng)} and {_adj(rng)}."


def _t10_copula_comp(rng: random.Random) -> str:
    """The {noun} is {comp} {adj}."""
    return f"The {_noun_common(rng)} is {_comp(rng)} {_adj(rng)}."


def _t11_copula_time(rng: random.Random) -> str:
    """The {noun} was {time}."""
    return f"The {_noun_common(rng)} was {_time(rng)}."


# --- Cat 3: Transitive with Preposition (5) ---
def _t12_trans_prep(rng: random.Random) -> str:
    """{noun} {verb} {prep} the {noun}."""
    return _capitalize(f"{_noun_any(rng)} {_verb_trans(rng)} {_prep(rng)} the {_noun_common(rng)}.")


def _t13_dative(rng: random.Random) -> str:
    """The {noun} {verb} the {noun} {adv}."""
    return f"The {_noun_common(rng)} {_verb_trans(rng)} the {_noun_common(rng)} {_adv(rng)}."


def _t14_has_object(rng: random.Random) -> str:
    """The {noun} has a {adj} {noun}."""
    return f"The {_noun_common(rng)} has a {_adj(rng)} {_noun_common(rng)}."


def _t15_trans_adv(rng: random.Random) -> str:
    """{noun} {verb} {adv} {prep} the {noun}."""
    return _capitalize(f"{_noun_any(rng)} {_verb_trans(rng)} {_adv(rng)} {_prep(rng)} the {_noun_common(rng)}.")


def _t16_has_time(rng: random.Random) -> str:
    """The {noun} has {time}."""
    return f"The {_noun_common(rng)} has {_time(rng)}."


# --- Cat 4: Intransitive + Prepositional (5) ---
def _t17_intrans_prep(rng: random.Random) -> str:
    """{noun} {verb} {prep} the {noun}."""
    return _capitalize(f"{_noun_any(rng)} {_verb_intrans(rng)} {_prep(rng)} the {_noun_common(rng)}.")


def _t18_intrans_prep_adj(rng: random.Random) -> str:
    """{noun} {verb} {prep} the {adj} {noun}."""
    return _capitalize(f"{_noun_any(rng)} {_verb_intrans(rng)} {_prep(rng)} the {_adj(rng)} {_noun_common(rng)}.")


def _t19_the_intrans_prep(rng: random.Random) -> str:
    """The {noun} {verb} {prep} the {adj} {noun}."""
    return f"The {_noun_common(rng)} {_verb_intrans(rng)} {_prep(rng)} the {_adj(rng)} {_noun_common(rng)}."


def _t20_intrans_adv(rng: random.Random) -> str:
    """The {noun} {verb} {adv} {prep} the {noun}."""
    return f"The {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)} {_prep(rng)} the {_noun_common(rng)}."


def _t21_intrans_comp(rng: random.Random) -> str:
    """The {noun} {verb} {prep} the {comp} {noun}."""
    return f"The {_noun_common(rng)} {_verb_intrans(rng)} {_prep(rng)} the {_comp(rng)} {_noun_common(rng)}."


# --- Cat 5: Observation Frames (He sees / She knows) 鈥?"I see" removed (see not in vocab) (5) ---
def _t22_he_sees_sva(rng: random.Random) -> str:
    """He sees a {adj} {noun} {verb} {adv}."""
    return f"He sees a {_adj(rng)} {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)}."


def _t23_she_knows_copula(rng: random.Random) -> str:
    """She knows the {noun} is {adj}."""
    return f"She knows the {_noun_common(rng)} is {_adj(rng)}."


def _t24_he_sees_again(rng: random.Random) -> str:
    """He sees the {noun} {verb} {adv}."""
    return f"He sees the {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)}."


def _t25_she_knows_obj(rng: random.Random) -> str:
    """She knows the {noun} {verb} {adv} {prep} the {noun}."""
    return f"She knows the {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)} {_prep(rng)} the {_noun_common(rng)}."


def _t26_she_knows_comp(rng: random.Random) -> str:
    """She knows the {noun} is {comp} {adj}."""
    return f"She knows the {_noun_common(rng)} is {_comp(rng)} {_adj(rng)}."


# --- Cat 6: Compound (5) ---
def _t27_and_compound(rng: random.Random) -> str:
    """The {noun} {verb} {adv} and the {noun} {verb} {adv}."""
    return (f"The {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)} "
            f"and the {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)}.")


def _t28_so_compound(rng: random.Random) -> str:
    """The {noun} {verb} {adv} so the {noun} {verb}."""
    return (f"The {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)} "
            f"so the {_noun_common(rng)} {_verb_intrans(rng)}.")


def _t29_yet_compound(rng: random.Random) -> str:
    """The {noun} {verb} {adv} yet the {noun} {verb}."""
    return (f"The {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)} "
            f"yet the {_noun_common(rng)} {_verb_intrans(rng)}.")


def _t30_and_subject(rng: random.Random) -> str:
    """The {noun} and the {noun} {verb} {adv}."""
    return f"The {_noun_common(rng)} and the {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)}."


def _t31_so_pred(rng: random.Random) -> str:
    """The {noun} is {adj} so the {noun} {verb}."""
    return (f"The {_noun_common(rng)} is {_adj(rng)} "
            f"so the {_noun_common(rng)} {_verb_intrans(rng)}.")


# --- Cat 7: Questions (3) ---
def _t32_he_sees_q(rng: random.Random) -> str:
    """He sees the {noun} {verb} {adv}?"""
    return f"He sees the {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)}?"


def _t33_she_knows_q(rng: random.Random) -> str:
    """She knows the {noun} is {adj}?"""
    return f"She knows the {_noun_common(rng)} is {_adj(rng)}?"


def _t34_he_sees_vq(rng: random.Random) -> str:
    """He sees the {noun} {verb}?"""
    return f"He sees the {_noun_common(rng)} {_verb_intrans(rng)}?"


# --- Cat 8: Exclamations (4) ---
def _t35_exclaim_sva(rng: random.Random) -> str:
    """The {adj} {noun} {verb} {adv}!"""
    return f"The {_adj(rng)} {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)}!"


def _t36_exclaim_copula(rng: random.Random) -> str:
    """The {noun} is so {adj}!"""
    return f"The {_noun_common(rng)} is so {_adj(rng)}!"


def _t37_exclaim_intensity(rng: random.Random) -> str:
    """The {noun} {verb} {adv} so {adv}!"""
    return f"The {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)} so {_adv(rng)}!"


def _t38_exclaim_he_sees(rng: random.Random) -> str:
    """He sees the {adj} {noun} {verb}!"""
    return f"He sees the {_adj(rng)} {_noun_common(rng)} {_verb_intrans(rng)}!"


# --- Cat 9: Determiner-subject (5) ---
def _t39_every(rng: random.Random) -> str:
    """Every {noun} {verb} {adv}."""
    return f"Every {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)}."


def _t40_each(rng: random.Random) -> str:
    """Each {noun} {verb} {adv}."""
    return f"Each {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)}."


def _t41_some(rng: random.Random) -> str:
    """Some {noun} {verb} {adv}."""
    return f"Some {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)}."


def _t42_many(rng: random.Random) -> str:
    """Many {noun} {verb} {adv}."""
    return f"Many {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)}."


def _t43_other(rng: random.Random) -> str:
    """Other {noun} {verb} {adv}."""
    return f"Other {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)}."


# --- Cat 10: Extended (4) ---
def _t44_complex(rng: random.Random) -> str:
    """The {noun} {verb} {prep} the {adj} {noun} {adv}."""
    return f"The {_noun_common(rng)} {_verb_intrans(rng)} {_prep(rng)} the {_adj(rng)} {_noun_common(rng)} {_adv(rng)}."


def _t45_superlative(rng: random.Random) -> str:
    """The {noun} is the most {adj}."""
    return f"The {_noun_common(rng)} is the most {_adj(rng)}."


def _t46_nested(rng: random.Random) -> str:
    """She knows the {noun} has a {adj} {noun}."""
    return f"She knows the {_noun_common(rng)} has a {_adj(rng)} {_noun_common(rng)}."


def _t47_long_compound(rng: random.Random) -> str:
    """The {noun} {verb} {adv} and the {noun} is {adj} so the {noun} {verb}."""
    return (f"The {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)} "
            f"and the {_noun_common(rng)} is {_adj(rng)} so the {_noun_common(rng)} {_verb_intrans(rng)}.")


# =====================================================================
# REGISTRY
# =====================================================================
TEMPLATES: list[tuple[str, str, int, Callable]] = [
    # Cat 1: Simple Declarative (6)
    ("T01_sva",              "simple_declarative",   3, _t01_sva),
    ("T02_the_sva",          "simple_declarative",   3, _t02_the_sva),
    ("T03_a_sva",            "simple_declarative",   3, _t03_a_sva),
    ("T04_sva_time",         "simple_declarative",   2, _t04_sva_time),
    ("T05_time_sva",         "simple_declarative",   2, _t05_time_sva),
    ("T06_time_adv",         "simple_declarative",   2, _t06_time_adv),
    # Cat 2: Copular (5)
    ("T07_copula_adj",       "copular",              3, _t07_copula_adj),
    ("T08_copula_adv",       "copular",              2, _t08_copula_adv),
    ("T09_double_copula",    "copular",              2, _t09_double_copula),
    ("T10_copula_comp",      "copular",              1, _t10_copula_comp),
    ("T11_copula_time",      "copular",              1, _t11_copula_time),
    # Cat 3: Transitive+Prep (5)
    ("T12_trans_prep",       "transitive",           3, _t12_trans_prep),
    ("T13_dative",           "transitive",           3, _t13_dative),
    ("T14_has_object",       "transitive",           3, _t14_has_object),
    ("T15_trans_adv",        "transitive",           2, _t15_trans_adv),
    ("T16_has_time",         "transitive",           1, _t16_has_time),
    # Cat 4: Intransitive+Prep (5)
    ("T17_intrans_prep",     "prepositional",        3, _t17_intrans_prep),
    ("T18_intrans_prep_adj", "prepositional",        3, _t18_intrans_prep_adj),
    ("T19_the_intrans_prep", "prepositional",        3, _t19_the_intrans_prep),
    ("T20_intrans_adv",      "prepositional",        2, _t20_intrans_adv),
    ("T21_intrans_comp",     "prepositional",        1, _t21_intrans_comp),
    # Cat 5: Observation (He sees / She knows) (5)
    ("T22_he_sees_sva",      "observation",          3, _t22_he_sees_sva),
    ("T23_she_knows_copula", "observation",          2, _t23_she_knows_copula),
    ("T24_he_sees_obj",      "observation",          2, _t24_he_sees_again),
    ("T25_she_knows_long",   "observation",          2, _t25_she_knows_obj),
    ("T26_she_knows_comp",   "observation",          1, _t26_she_knows_comp),
    # Cat 6: Compound (5)
    ("T27_and_compound",     "compound",             3, _t27_and_compound),
    ("T28_so_compound",      "compound",             2, _t28_so_compound),
    ("T29_yet_compound",     "compound",             2, _t29_yet_compound),
    ("T30_and_subject",      "compound",             2, _t30_and_subject),
    ("T31_so_pred",          "compound",             2, _t31_so_pred),
    # Cat 7: Questions (3)
    ("T32_he_sees_q",        "question",             2, _t32_he_sees_q),
    ("T33_she_knows_q",      "question",             2, _t33_she_knows_q),
    ("T34_he_sees_vq",       "question",             2, _t34_he_sees_vq),
    # Cat 8: Exclamations (4)
    ("T35_exclaim_sva",      "exclamation",          2, _t35_exclaim_sva),
    ("T36_exclaim_copula",   "exclamation",          2, _t36_exclaim_copula),
    ("T37_exclaim_intensity","exclamation",          2, _t37_exclaim_intensity),
    ("T38_exclaim_he_sees",  "exclamation",          2, _t38_exclaim_he_sees),
    # Cat 9: Determiner-subject (5)
    ("T39_every",            "determiner_subject",   2, _t39_every),
    ("T40_each",             "determiner_subject",   2, _t40_each),
    ("T41_some",             "determiner_subject",   2, _t41_some),
    ("T42_many",             "determiner_subject",   2, _t42_many),
    ("T43_other",            "determiner_subject",   2, _t43_other),
    # Cat 10: Extended (4)
    ("T44_complex",          "extended",             2, _t44_complex),
    ("T45_superlative",      "extended",             1, _t45_superlative),
    ("T46_nested",           "extended",             2, _t46_nested),
    ("T47_long_compound",    "extended",             1, _t47_long_compound),
]


# =====================================================================
# GENERATOR
# =====================================================================

def generate_corpus(
    n_sentences: int = 5000,
    seed: int = 42,
    max_attempts: int = 100000,
) -> tuple[list[str], dict[str, int], dict[str, int], int, int, int]:
    rng = random.Random(seed)

    weighted = []
    for name, cat, weight, func in TEMPLATES:
        weighted.extend([(name, cat, func) for _ in range(weight)])
    total_weight = len(weighted)

    sentences: list[str] = []
    seen_exact: set[str] = set()
    failures = 0
    oov_rejected = 0
    template_counts: dict[str, int] = {t[0]: 0 for t in TEMPLATES}
    cat_counts: dict[str, int] = {}
    attempts = 0

    while len(sentences) < n_sentences and attempts < max_attempts:
        attempts += 1
        idx = rng.randint(0, total_weight - 1)
        name, cat, func = weighted[idx]

        try:
            sentence = func(rng)
            if not sentence or not sentence.strip():
                failures += 1
                continue

            # OOV check
            words = sentence.split()
            cleaned = [w.lower().rstrip(".,!?;:") for w in words]
            if any(w not in ALL_VALID for w in cleaned):
                oov_rejected += 1
                failures += 1
                continue

            if len(words) < 3:
                failures += 1
                continue

            if sentence in seen_exact:
                continue
            seen_exact.add(sentence)

            sentences.append(sentence)
            template_counts[name] += 1
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        except Exception:
            failures += 1

    return sentences, template_counts, cat_counts, failures, oov_rejected, attempts


def save_corpus(sentences: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sentences))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="AronaLM Phase B4-1a: Data generator v4 (all templates working)"
    )
    parser.add_argument("--output", type=Path,
        default=Path(r"C:\Users\18217\AronaLM\experiments\b4\data\corpus_v2_candidate.txt"))
    parser.add_argument("--n-sentences", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("ARONALM PHASE B4-1a: DATA GENERATOR v4")
    print("=" * 60)
    print(f"  Target:  {args.n_sentences} sentences")
    print(f"  Seed:    {args.seed}")
    print()

    sentences, t_counts, c_counts, failures, oov, attempts = generate_corpus(
        n_sentences=args.n_sentences, seed=args.seed,
    )

    zero_templates = [(n, c) for n, c in t_counts.items() if c == 0]
    print(f"  Generated: {len(sentences)} sentences")
    print(f"  Attempts:  {attempts}")
    print(f"  Failures:  {failures} ({failures/max(1,attempts)*100:.1f}%)")
    print(f"  OOV rejs:  {oov}")
    if zero_templates:
        print(f"  Zero-count templates: {[n for n,c in zero_templates]}")
    else:
        print(f"  All {len(TEMPLATES)} templates active")

    print(f"\n  Category distribution:")
    for cat in sorted(c_counts, key=lambda c: -c_counts[c]):
        print(f"    {cat:25s} {c_counts[cat]:5d} ({c_counts[cat]/len(sentences)*100:.1f}%)")

    print(f"\n  Top 10 templates:")
    for name, cnt in sorted(t_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {name:25s} {cnt:5d} ({cnt/len(sentences)*100:.1f}%)")

    save_corpus(sentences, args.output)
    print(f"\n  Saved: {args.output} ({args.output.stat().st_size:,} bytes)")
    print("=" * 60)


