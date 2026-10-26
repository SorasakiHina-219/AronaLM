"""
Phase B4-3: Data generator with expanded vocabulary (B4-2 extension).

Uses experiments/b4/vocab_expand/vocab_extended.json (11 new structural words).
Templates cover: negation, questions, conditionals, causality, contrast, comparison.

Key vocabulary additions (IDs 558-568):
  not(558) no(559) what(560) where(561) how(562)
  if(563) when(564) but(565) so(566) than(567) because(568)

NOTE: "does"/"to"/"of"/"for"/"with"/"from"/"are"/"were"/"have" remain OOV.
      Question frames use "is"/"was" copula only.
      Negation uses "is not" / "has no" patterns.
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

CONJUNCTIONS: list[str] = ["and", "so", "yet", "but"]

DETERMINERS_LOWER: list[str] = [
    "the", "a", "some", "many",
    "each", "every", "other", "such",
]

COMPARATORS: list[str] = ["more", "most", "less"]

TIME_WORDS: list[str] = ["day", "night"]

# ===== NEW STRUCTURAL WORDS (from B4-2 extension) =====
NEGATION: list[str] = ["not", "no"]
QUESTIONS: list[str] = ["what", "where", "when", "how"]
CONDITIONAL: list[str] = ["if", "when"]
CAUSAL: list[str] = ["because", "so"]
COMPARATIVE: list[str] = ["than", "more", "most", "less"]

ALL_VALID: set[str] = set(
    ADJECTIVES + VERBS_INTRANSITIVE + VERBS_TRANSITIVE + VERBS_COPLA
    + VERBS_POSSESS + NOUNS_COMMON + PRONOUNS + ADVERBS
    + PREPOSITIONS + CONJUNCTIONS + DETERMINERS_LOWER
    + COMPARATORS + TIME_WORDS + NEGATION + QUESTIONS
    + CONDITIONAL + CAUSAL + COMPARATIVE
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


def _pronoun(rng: random.Random) -> str:
    return _r(rng, PRONOUNS)


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


def _neg(rng: random.Random) -> str:
    return _r(rng, NEGATION)


def _qword(rng: random.Random) -> str:
    return _r(rng, QUESTIONS)


def _cond(rng: random.Random) -> str:
    return _r(rng, CONDITIONAL)


def _causal(rng: random.Random) -> str:
    return _r(rng, CAUSAL)


def _capitalize(s: str) -> str:
    if s and s[0].islower():
        return s[0].upper() + s[1:]
    return s


# =====================================================================
# TEMPLATES — Category 1: Simple Declarative (6)
# =====================================================================

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


# =====================================================================
# TEMPLATES — Category 2: Copular (5)
# =====================================================================

def _t07_copula_adj(rng: random.Random) -> str:
    """The {noun} is {adj}."""
    return f"The {_noun_common(rng)} is {_adj(rng)}."


def _t08_copula_comp(rng: random.Random) -> str:
    """The {noun} is {comp} {adj}."""
    return f"The {_noun_common(rng)} is {_comp(rng)} {_adj(rng)}."


def _t09_double_copula(rng: random.Random) -> str:
    """The {noun} is {adj} and {adj}."""
    return f"The {_noun_common(rng)} is {_adj(rng)} and {_adj(rng)}."


def _t10_copula_time(rng: random.Random) -> str:
    """The {noun} was {time}."""
    return f"The {_noun_common(rng)} was {_time(rng)}."


def _t11_copula_adv(rng: random.Random) -> str:
    """The {noun} is {adv}. (avoid: adv after copula is marginal)"""
    return f"The {_noun_common(rng)} is {_adv(rng)}."


# =====================================================================
# TEMPLATES — Category 3: Transitive (5)
# =====================================================================

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


# =====================================================================
# TEMPLATES — Category 4: Prepositional (5)
# =====================================================================

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


# =====================================================================
# TEMPLATES — Category 5: Observation Frames (5)
# =====================================================================

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


# =====================================================================
# TEMPLATES — Category 6: Compound (5)
# =====================================================================

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


# =====================================================================
# TEMPLATES — Category 7: Negation (5) — NEW
# =====================================================================

def _t32_neg_copula(rng: random.Random) -> str:
    """The {noun} is not {adj}."""
    return f"The {_noun_common(rng)} is not {_adj(rng)}."


def _t33_neg_copula_adv(rng: random.Random) -> str:
    """The {noun} is not {adv}."""
    return f"The {_noun_common(rng)} is not {_adv(rng)}."


def _t34_neg_possess(rng: random.Random) -> str:
    """The {noun} has no {adj} {noun}."""
    return f"The {_noun_common(rng)} has no {_adj(rng)} {_noun_common(rng)}."


def _t35_neg_intrans(rng: random.Random) -> str:
    """The {noun} is not {adj} and {noun} {verb} {adv}."""
    n1 = _noun_common(rng)
    n2 = _noun_common(rng)
    return f"The {n1} is not {_adj(rng)} and {n2} {_verb_intrans(rng)} {_adv(rng)}."


def _t36_neg_observation(rng: random.Random) -> str:
    """He sees the {noun} is not {adj}."""
    return f"He sees the {_noun_common(rng)} is not {_adj(rng)}."


# =====================================================================
# TEMPLATES — Category 8: Questions (5) — NEW
# =====================================================================

def _t37_q_what_copula(rng: random.Random) -> str:
    """What is the {noun}?"""
    return f"What is the {_noun_common(rng)}?"


def _t38_q_where_copula(rng: random.Random) -> str:
    """Where is the {noun}?"""
    return f"Where is the {_noun_common(rng)}?"


def _t39_q_when_copula(rng: random.Random) -> str:
    """When is the {noun}?"""
    return f"When is the {_noun_common(rng)}?"


def _t40_q_how_copula(rng: random.Random) -> str:
    """How is the {noun}?"""
    return f"How is the {_noun_common(rng)}?"


def _t41_q_what_observation(rng: random.Random) -> str:
    """What sees the {noun}?"""
    return f"What sees the {_noun_common(rng)}?"


# =====================================================================
# TEMPLATES — Category 9: Conditionals (5) — NEW
# =====================================================================

def _t42_cond_if(rng: random.Random) -> str:
    """If the {noun} {verb} the {noun} {verb}."""
    return (f"If the {_noun_common(rng)} {_verb_intrans(rng)} "
            f"the {_noun_common(rng)} {_verb_intrans(rng)}.")


def _t43_cond_when(rng: random.Random) -> str:
    """When the {noun} {verb} the {noun} {verb}."""
    return (f"When the {_noun_common(rng)} {_verb_intrans(rng)} "
            f"the {_noun_common(rng)} {_verb_intrans(rng)}.")


def _t44_cond_if_copula(rng: random.Random) -> str:
    """If the {noun} is {adj} the {noun} {verb}."""
    return (f"If the {_noun_common(rng)} is {_adj(rng)} "
            f"the {_noun_common(rng)} {_verb_intrans(rng)}.")


def _t45_cond_when_time(rng: random.Random) -> str:
    """When the {time} the {noun} {verb} {adv}."""
    return (f"When the {_time(rng)} "
            f"the {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)}.")


def _t46_cond_if_observation(rng: random.Random) -> str:
    """If the {noun} is {adj} he sees the {noun}."""
    return (f"If the {_noun_common(rng)} is {_adj(rng)} "
            f"He sees the {_noun_common(rng)}.")


# =====================================================================
# TEMPLATES — Category 10: Causality (4) — NEW
# =====================================================================

def _t47_causal_because(rng: random.Random) -> str:
    """The {noun} {verb} {adv} because the {noun} is {adj}."""
    return (f"The {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)} "
            f"because the {_noun_common(rng)} is {_adj(rng)}.")


def _t48_causal_so_copula(rng: random.Random) -> str:
    """The {noun} is {adj} so the {noun} {verb}."""
    return (f"The {_noun_common(rng)} is {_adj(rng)} "
            f"so the {_noun_common(rng)} {_verb_intrans(rng)}.")


def _t49_causal_because_copula(rng: random.Random) -> str:
    """The {noun} is {adj} because the {noun} {verb} {adv}."""
    return (f"The {_noun_common(rng)} is {_adj(rng)} "
            f"because the {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)}.")


def _t50_causal_so_compound(rng: random.Random) -> str:
    """The {noun} {verb} {adv} so the {noun} is {adj}."""
    return (f"The {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)} "
            f"so the {_noun_common(rng)} is {_adj(rng)}.")


# =====================================================================
# TEMPLATES — Category 11: Contrast / but (4) — NEW
# =====================================================================

def _t51_contrast_but(rng: random.Random) -> str:
    """The {noun} is {adj} but the {noun} is {adj}."""
    return (f"The {_noun_common(rng)} is {_adj(rng)} "
            f"but the {_noun_common(rng)} is {_adj(rng)}.")


def _t52_contrast_but_adv(rng: random.Random) -> str:
    """The {noun} is {adj} but {noun} {verb} {adv}."""
    return (f"The {_noun_common(rng)} is {_adj(rng)} "
            f"but {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)}.")


def _t53_contrast_yet(rng: random.Random) -> str:
    """The {noun} is {adj} yet the {noun} {verb} {adv}."""
    return (f"The {_noun_common(rng)} is {_adj(rng)} "
            f"yet the {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)}.")


def _t54_contrast_but_observation(rng: random.Random) -> str:
    """He sees the {noun} is {adj} but the {noun} is {adj}."""
    return (f"He sees the {_noun_common(rng)} is {_adj(rng)} "
            f"but the {_noun_common(rng)} is {_adj(rng)}.")


# =====================================================================
# TEMPLATES — Category 12: Comparison / than (4) — NEW
# =====================================================================

def _t55_comp_more_than(rng: random.Random) -> str:
    """The {noun} is more {adj} than the {noun}."""
    return (f"The {_noun_common(rng)} is more {_adj(rng)} "
            f"than the {_noun_common(rng)}.")


def _t56_comp_most(rng: random.Random) -> str:
    """The {noun} is the most {adj}."""
    return f"The {_noun_common(rng)} is the most {_adj(rng)}."


def _t57_comp_less_than(rng: random.Random) -> str:
    """The {noun} is less {adj} than the {noun}."""
    return (f"The {_noun_common(rng)} is less {_adj(rng)} "
            f"than the {_noun_common(rng)}.")


def _t58_comp_more_observation(rng: random.Random) -> str:
    """She knows the {noun} is more {adj} than the {noun}."""
    return (f"She knows the {_noun_common(rng)} is more {_adj(rng)} "
            f"than the {_noun_common(rng)}.")


# =====================================================================
# TEMPLATES — Category 13: Exclamation (4)
# =====================================================================

def _t59_exclaim_sva(rng: random.Random) -> str:
    """The {adj} {noun} {verb} {adv}!"""
    return f"The {_adj(rng)} {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)}!"


def _t60_exclaim_copula(rng: random.Random) -> str:
    """The {noun} is so {adj}!"""
    return f"The {_noun_common(rng)} is so {_adj(rng)}!"


def _t61_exclaim_intensity(rng: random.Random) -> str:
    """The {noun} {verb} {adv} so {adv}!"""
    return f"The {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)} so {_adv(rng)}!"


def _t62_exclaim_he_sees(rng: random.Random) -> str:
    """He sees the {adj} {noun} {verb}!"""
    return f"He sees the {_adj(rng)} {_noun_common(rng)} {_verb_intrans(rng)}!"


# =====================================================================
# TEMPLATES — Category 14: Determiner-subject (5)
# =====================================================================

def _t63_every(rng: random.Random) -> str:
    """Every {noun} {verb} {adv}."""
    return f"Every {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)}."


def _t64_each(rng: random.Random) -> str:
    """Each {noun} {verb} {adv}."""
    return f"Each {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)}."


def _t65_some(rng: random.Random) -> str:
    """Some {noun} {verb} {adv}."""
    return f"Some {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)}."


def _t66_many(rng: random.Random) -> str:
    """Many {noun} {verb} {adv}."""
    return f"Many {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)}."


def _t67_other(rng: random.Random) -> str:
    """Other {noun} {verb} {adv}."""
    return f"Other {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)}."


# =====================================================================
# TEMPLATES — Category 15: Extended (4)
# =====================================================================

def _t68_complex(rng: random.Random) -> str:
    """The {noun} {verb} {prep} the {adj} {noun} {adv}."""
    return f"The {_noun_common(rng)} {_verb_intrans(rng)} {_prep(rng)} the {_adj(rng)} {_noun_common(rng)} {_adv(rng)}."


def _t69_nested(rng: random.Random) -> str:
    """She knows the {noun} has a {adj} {noun}."""
    return f"She knows the {_noun_common(rng)} has a {_adj(rng)} {_noun_common(rng)}."


def _t70_long_compound(rng: random.Random) -> str:
    """The {noun} {verb} {adv} and the {noun} is {adj} so the {noun} {verb}."""
    return (f"The {_noun_common(rng)} {_verb_intrans(rng)} {_adv(rng)} "
            f"and the {_noun_common(rng)} is {_adj(rng)} so the {_noun_common(rng)} {_verb_intrans(rng)}.")


def _t71_complex_observation(rng: random.Random) -> str:
    """He sees the {adj} {noun} {verb} {prep} the {adj} {noun}."""
    return (f"He sees the {_adj(rng)} {_noun_common(rng)} {_verb_intrans(rng)} "
            f"{_prep(rng)} the {_adj(rng)} {_noun_common(rng)}.")


# =====================================================================
# REGISTRY
# =====================================================================
# Format: (name, category, weight, function)
TEMPLATES: list[tuple[str, str, int, Callable]] = [
    # Cat 1: Simple Declarative (6) — weight 2 each = 12 total
    ("T01_sva",             "simple_declarative",   2, _t01_sva),
    ("T02_the_sva",         "simple_declarative",   2, _t02_the_sva),
    ("T03_a_sva",           "simple_declarative",   2, _t03_a_sva),
    ("T04_sva_time",        "simple_declarative",   2, _t04_sva_time),
    ("T05_time_sva",        "simple_declarative",   2, _t05_time_sva),
    ("T06_time_adv",        "simple_declarative",   2, _t06_time_adv),
    # Cat 2: Copular (5) — weight 2 each = 10 total
    ("T07_copula_adj",      "copular",              2, _t07_copula_adj),
    ("T08_copula_comp",     "copular",              2, _t08_copula_comp),
    ("T09_double_copula",   "copular",              2, _t09_double_copula),
    ("T10_copula_time",     "copular",              2, _t10_copula_time),
    ("T11_copula_adv",      "copular",              1, _t11_copula_adv),
    # Cat 3: Transitive (5) — weight 2 each = 10 total
    ("T12_trans_prep",      "transitive",           2, _t12_trans_prep),
    ("T13_dative",          "transitive",           2, _t13_dative),
    ("T14_has_object",      "transitive",           2, _t14_has_object),
    ("T15_trans_adv",       "transitive",           2, _t15_trans_adv),
    ("T16_has_time",        "transitive",           1, _t16_has_time),
    # Cat 4: Prepositional (5) — weight 2 each = 10 total
    ("T17_intrans_prep",    "prepositional",        2, _t17_intrans_prep),
    ("T18_intrans_prep_adj","prepositional",        2, _t18_intrans_prep_adj),
    ("T19_the_intrans_prep","prepositional",        2, _t19_the_intrans_prep),
    ("T20_intrans_adv",     "prepositional",        2, _t20_intrans_adv),
    ("T21_intrans_comp",    "prepositional",        1, _t21_intrans_comp),
    # Cat 5: Observation (5) — weight 2 each = 10 total
    ("T22_he_sees_sva",     "observation",          2, _t22_he_sees_sva),
    ("T23_she_knows_copula","observation",          2, _t23_she_knows_copula),
    ("T24_he_sees_again",   "observation",          2, _t24_he_sees_again),
    ("T25_she_knows_obj",   "observation",          2, _t25_she_knows_obj),
    ("T26_she_knows_comp",  "observation",          1, _t26_she_knows_comp),
    # Cat 6: Compound (5) — weight 2 each = 10 total
    ("T27_and_compound",    "compound",             2, _t27_and_compound),
    ("T28_so_compound",     "compound",             2, _t28_so_compound),
    ("T29_yet_compound",    "compound",             2, _t29_yet_compound),
    ("T30_and_subject",     "compound",             2, _t30_and_subject),
    ("T31_so_pred",         "compound",             2, _t31_so_pred),
    # Cat 7: Negation (5) — weight 3 each = 15 total
    ("T32_neg_copula",      "negation",             3, _t32_neg_copula),
    ("T33_neg_copula_adv",  "negation",             2, _t33_neg_copula_adv),
    ("T34_neg_possess",     "negation",             3, _t34_neg_possess),
    ("T35_neg_intrans",     "negation",             2, _t35_neg_intrans),
    ("T36_neg_observation", "negation",             2, _t36_neg_observation),
    # Cat 8: Questions (5) — weight 3 each = 15 total
    ("T37_q_what_copula",   "question",             3, _t37_q_what_copula),
    ("T38_q_where_copula",  "question",             3, _t38_q_where_copula),
    ("T39_q_when_copula",   "question",             3, _t39_q_when_copula),
    ("T40_q_how_copula",    "question",             3, _t40_q_how_copula),
    ("T41_q_what_obs",      "question",             2, _t41_q_what_observation),
    # Cat 9: Conditionals (5) — weight 3 each = 15 total
    ("T42_cond_if",         "conditional",          3, _t42_cond_if),
    ("T43_cond_when",       "conditional",          3, _t43_cond_when),
    ("T44_cond_if_copula",  "conditional",          2, _t44_cond_if_copula),
    ("T45_cond_when_time",  "conditional",          2, _t45_cond_when_time),
    ("T46_cond_if_obs",     "conditional",          2, _t46_cond_if_observation),
    # Cat 10: Causality (4) — weight 3 each = 12 total
    ("T47_causal_because",  "causality",            3, _t47_causal_because),
    ("T48_causal_so",       "causality",            3, _t48_causal_so_copula),
    ("T49_causal_bec_cop",  "causality",            2, _t49_causal_because_copula),
    ("T50_causal_so_adj",   "causality",            2, _t50_causal_so_compound),
    # Cat 11: Contrast (4) — weight 3 each = 12 total
    ("T51_contrast_but",    "contrast",             3, _t51_contrast_but),
    ("T52_contrast_but_adv","contrast",             2, _t52_contrast_but_adv),
    ("T53_contrast_yet",    "contrast",             2, _t53_contrast_yet),
    ("T54_contrast_but_obs","contrast",             2, _t54_contrast_but_observation),
    # Cat 12: Comparison (4) — weight 3 each = 12 total
    ("T55_comp_more_than",  "comparison",           3, _t55_comp_more_than),
    ("T56_comp_most",       "comparison",           2, _t56_comp_most),
    ("T57_comp_less_than",  "comparison",           2, _t57_comp_less_than),
    ("T58_comp_more_obs",   "comparison",           2, _t58_comp_more_observation),
    # Cat 13: Exclamation (4) — weight 2 each = 8 total
    ("T59_exclaim_sva",     "exclamation",          2, _t59_exclaim_sva),
    ("T60_exclaim_copula",  "exclamation",          2, _t60_exclaim_copula),
    ("T61_exclaim_intens",  "exclamation",          2, _t61_exclaim_intensity),
    ("T62_exclaim_he_sees", "exclamation",          2, _t62_exclaim_he_sees),
    # Cat 14: Determiner-subject (5) — weight 2 each = 10 total
    ("T63_every",           "determiner_subject",   2, _t63_every),
    ("T64_each",            "determiner_subject",   2, _t64_each),
    ("T65_some",            "determiner_subject",   2, _t65_some),
    ("T66_many",            "determiner_subject",   2, _t66_many),
    ("T67_other",           "determiner_subject",   2, _t67_other),
    # Cat 15: Extended (4) — weight 2 each = 8 total
    ("T68_complex",         "extended",             2, _t68_complex),
    ("T69_nested",          "extended",             2, _t69_nested),
    ("T70_long_compound",   "extended",             2, _t70_long_compound),
    ("T71_complex_obs",     "extended",             2, _t71_complex_observation),
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
        description="AronaLM Phase B4-3: Data generator v3 (expanded vocab)"
    )
    parser.add_argument("--output", type=Path,
        default=Path(r"C:\Users\18217\AronaLM\experiments\b4\b4_3\corpus_v3_candidate.txt"))
    parser.add_argument("--n-sentences", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("ARONALM PHASE B4-3: DATA GENERATOR v3")
    print("=" * 60)
    print(f"  Target:  {args.n_sentences} sentences")
    print(f"  Seed:    {args.seed}")
    print(f"  Templates: {len(TEMPLATES)} across 15 categories")
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

    print(f"\n  Top 15 templates:")
    for name, cnt in sorted(t_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"    {name:25s} {cnt:5d} ({cnt/len(sentences)*100:.1f}%)")

    save_corpus(sentences, args.output)
    print(f"\n  Saved: {args.output} ({args.output.stat().st_size:,} bytes)")
    print("=" * 60)
