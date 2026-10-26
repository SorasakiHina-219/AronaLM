"""
Phase B4-1a: Quality audit script for candidate corpus v2.

Reads the generated candidate corpus and compares quality metrics
against the baseline (original corpus). All metrics computed consistently.
"""

import sys
sys.path.insert(0, r"C:\Users\18217\AronaLM")

from pathlib import Path
from collections import Counter
import math
import json

root = Path(r"C:\Users\18217\AronaLM")

# =====================================================================
# LOAD DATA
# =====================================================================
print("=" * 60)
print("PHASE B4-1a: QUALITY AUDIT")
print("=" * 60)

# Baseline (original corpus)
baseline_path = root / "data" / "raw" / "corpus.txt"
baseline_lines = [l.strip() for l in baseline_path.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"\n  Baseline corpus: {len(baseline_lines)} sentences")

# Candidate (new corpus)
candidate_path = root / "experiments" / "b4" / "data" / "corpus_v2_candidate.txt"
if not candidate_path.exists():
    print(f"  ERROR: Candidate corpus not found at {candidate_path}")
    sys.exit(1)
candidate_lines = [l.strip() for l in candidate_path.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"  Candidate corpus: {len(candidate_lines)} sentences")

# Existing splits for cross-check
train_lines = [l.strip() for l in (root / "data" / "splits_v2" / "train.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
val_lines = [l.strip() for l in (root / "data" / "splits_v2" / "val.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
test_lines = [l.strip() for l in (root / "data" / "splits_v2" / "test.txt").read_text(encoding="utf-8").splitlines() if l.strip()]

# Tokenizer
from aronalm.tokenizer import BPETokenizer
tokenizer = BPETokenizer.load(root / "tokenizer" / "vocab.json")


def word_count(s):
    """Count whitespace-separated words in a sentence."""
    return len(s.split())


def normalize(s):
    """Normalize for dedup: lowercase, strip punctuation, collapse whitespace."""
    import re
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    s = s.rstrip('.,!?;:')
    return s


def compute_2gram_dup_rate(sentences):
    """Compute 2-gram duplicate rate."""
    ngrams = []
    for s in sentences:
        ws = s.split()
        for i in range(len(ws) - 1):
            ngrams.append((ws[i], ws[i+1]))
    ng_count = Counter(ngrams)
    total = len(ngrams)
    dup = sum(c - 1 for c in ng_count.values() if c > 1)
    return total, dup, dup / total if total > 0 else 0


def compute_3gram_dup_rate(sentences):
    """Compute 3-gram duplicate rate."""
    ngrams = []
    for s in sentences:
        ws = s.split()
        for i in range(len(ws) - 2):
            ngrams.append(tuple(ws[i:i+3]))
    ng_count = Counter(ngrams)
    total = len(ngrams)
    dup = sum(c - 1 for c in ng_count.values() if c > 1)
    return total, dup, dup / total if total > 0 else 0


def compute_top3_first_word_concentration(sentences):
    """Compute top-3 first word concentration."""
    first_words = Counter(s.split()[0] for s in sentences)
    total = len(sentences)
    top3 = sum(c for _, c in first_words.most_common(3))
    return top3 / total


def compute_herdans_c(sentences):
    """Compute Herdan's C = log(V) / log(N)."""
    words = []
    for s in sentences:
        words.extend(s.split())
    n = len(words)
    v = len(set(words))
    if n <= 1:
        return 0.0
    return math.log(v) / math.log(n)


def compute_vocabulary_utilization(sentences):
    """Compute fraction of vocab tokens used in sentences."""
    all_tokens = set()
    for s in sentences:
        for word in s.split():
            all_tokens.add(word.lower().rstrip('.,!?;:'))
    vocab_set = set(tokenizer.vocab.keys())
    # Check which words are in vocab
    covered = sum(1 for w in all_tokens if w in vocab_set)
    return covered / len(all_tokens) if all_tokens else 0


def compute_sentence_length_stats(sentences):
    """Compute sentence length statistics."""
    lens = [word_count(s) for s in sentences]
    if not lens:
        return {}
    import statistics
    return {
        "min": min(lens),
        "max": max(lens),
        "mean": statistics.mean(lens),
        "stdev": statistics.stdev(lens) if len(lens) > 1 else 0,
        "median": statistics.median(lens),
    }


def compute_punctuation_distribution(sentences):
    """Compute distribution of sentence-ending punctuation."""
    periods = sum(1 for s in sentences if s.endswith('.'))
    question_marks = sum(1 for s in sentences if s.endswith('?'))
    exclamation = sum(1 for s in sentences if s.endswith('!'))
    total = len(sentences)
    return {
        "period": periods / total * 100 if total else 0,
        "question": question_marks / total * 100 if total else 0,
        "exclamation": exclamation / total * 100 if total else 0,
    }


def compute_complex_ratio(sentences):
    """Compute ratio of compound/complex sentences."""
    complex_count = 0
    for s in sentences:
        if any(c in s for c in [' and ', ' so ', ' yet ', '?', '!']):
            complex_count += 1
    return complex_count / len(sentences) * 100 if sentences else 0


def compute_negation_ratio(sentences):
    """Compute ratio of sentences containing negation indicators."""
    neg_count = 0
    for s in sentences:
        if ' not ' in s.lower() or ' no ' in s.lower():
            neg_count += 1
    return neg_count / len(sentences) * 100 if sentences else 0


def compute_template_coverage(sentences):
    """Estimate template coverage by analyzing sentence patterns."""
    patterns = Counter()
    for s in sentences:
        ws = s.split()
        if len(ws) >= 2:
            # Pattern: first 2 words
            pat = tuple(ws[:2])
            patterns[pat] += 1
    total = len(sentences)
    unique_patterns = len(patterns)
    top1_pct = patterns.most_common(1)[0][1] / total * 100 if patterns else 0
    top5_pct = sum(c for _, c in patterns.most_common(5)) / total * 100
    return {
        "unique_patterns_2word": unique_patterns,
        "top1_pattern_pct": top1_pct,
        "top5_patterns_pct": top5_pct,
    }


def compute_cross_set_overlap(candidate_sentences, ref_sets):
    """Check sentence-level overlap with reference sets."""
    candidate_set = set(candidate_sentences)
    overlaps = {}
    for name, ref in ref_sets.items():
        ref_set = set(ref)
        overlap = candidate_set & ref_set
        overlaps[name] = {
            "count": len(overlap),
            "pct_of_candidate": len(overlap) / len(candidate_set) * 100 if candidate_set else 0,
            "pct_of_ref": len(overlap) / len(ref_set) * 100 if ref_set else 0,
        }
        if overlap:
            overlaps[name]["examples"] = list(overlap)[:5]
    return overlaps


def compute_tokenizer_coverage(sentences):
    """Check how many unique words from sentences are in the tokenizer vocab."""
    all_words = set()
    for s in sentences:
        for word in s.split():
            cleaned = word.lower().rstrip('.,!?;:')
            all_words.add(cleaned)
    vocab_words = set(tokenizer.vocab.keys())
    missing = all_words - vocab_words
    return {
        "total_unique_words": len(all_words),
        "in_vocab": len(all_words & vocab_words),
        "missing_from_vocab": len(missing),
        "missing_words": sorted(missing)[:20],
    }


# =====================================================================
# COMPUTE METRICS
# =====================================================================
print("\n" + "=" * 60)
print("QUALITY METRICS")
print("=" * 60)

results = {}

# 1. Unique sentence ratio
baseline_unique = len(set(baseline_lines))
candidate_unique = len(set(candidate_lines))
results["unique_ratio_baseline"] = baseline_unique / len(baseline_lines) * 100
results["unique_ratio_candidate"] = candidate_unique / len(candidate_lines) * 100
print(f"\n[1] Unique sentence ratio:")
print(f"    Baseline:    {baseline_unique}/{len(baseline_lines)} = {results['unique_ratio_baseline']:.1f}%")
print(f"    Candidate:   {candidate_unique}/{len(candidate_lines)} = {results['unique_ratio_candidate']:.1f}%")

# 2. N-gram duplication
baseline_2g_total, baseline_2g_dup, baseline_2g_rate = compute_2gram_dup_rate(baseline_lines)
candidate_2g_total, candidate_2g_dup, candidate_2g_rate = compute_2gram_dup_rate(candidate_lines)
baseline_3g_total, baseline_3g_dup, baseline_3g_rate = compute_3gram_dup_rate(baseline_lines)
candidate_3g_total, candidate_3g_dup, candidate_3g_rate = compute_3gram_dup_rate(candidate_lines)
results["2gram_rate_baseline"] = baseline_2g_rate
results["2gram_rate_candidate"] = candidate_2g_rate
results["3gram_rate_baseline"] = baseline_3g_rate
results["3gram_rate_candidate"] = candidate_3g_rate
print(f"\n[2] N-gram duplication rate:")
print(f"    2-gram baseline:    {baseline_2g_rate*100:.1f}% ({baseline_2g_dup}/{baseline_2g_total})")
print(f"    2-gram candidate:   {candidate_2g_rate*100:.1f}% ({candidate_2g_dup}/{candidate_2g_total})")
print(f"    3-gram baseline:    {baseline_3g_rate*100:.1f}%")
print(f"    3-gram candidate:   {candidate_3g_rate*100:.1f}%")

# 3. Top-3 first word concentration
baseline_top3 = compute_top3_first_word_concentration(baseline_lines)
candidate_top3 = compute_top3_first_word_concentration(candidate_lines)
results["top3_first_baseline"] = baseline_top3
results["top3_first_candidate"] = candidate_top3
print(f"\n[3] Top-3 first word concentration:")
print(f"    Baseline:    {baseline_top3*100:.1f}%")
print(f"    Candidate:   {candidate_top3*100:.1f}%")

# 4. Vocabulary utilization
baseline_vocab = compute_vocabulary_utilization(baseline_lines)
candidate_vocab = compute_vocabulary_utilization(candidate_lines)
results["vocab_util_baseline"] = baseline_vocab
results["vocab_util_candidate"] = candidate_vocab
print(f"\n[4] Vocabulary utilization (unique words in sentence / in vocab):")
print(f"    Baseline:    {baseline_vocab*100:.1f}%")
print(f"    Candidate:   {candidate_vocab*100:.1f}%")

# 5. Herdan's C
baseline_herdan = compute_herdans_c(baseline_lines)
candidate_herdan = compute_herdans_c(candidate_lines)
results["herdan_baseline"] = baseline_herdan
results["herdan_candidate"] = candidate_herdan
print(f"\n[5] Herdan's C (log(V)/log(N)):")
print(f"    Baseline:    {baseline_herdan:.4f}")
print(f"    Candidate:   {candidate_herdan:.4f}")

# 6. Sentence length stats
baseline_lens = compute_sentence_length_stats(baseline_lines)
candidate_lens = compute_sentence_length_stats(candidate_lines)
results["length_baseline"] = baseline_lens
results["length_candidate"] = candidate_lens
print(f"\n[6] Sentence length statistics:")
print(f"    Baseline:    min={baseline_lens['min']} max={baseline_lens['max']} mean={baseline_lens['mean']:.1f} stdev={baseline_lens['stdev']:.2f} median={baseline_lens['median']}")
print(f"    Candidate:   min={candidate_lens['min']} max={candidate_lens['max']} mean={candidate_lens['mean']:.1f} stdev={candidate_lens['stdev']:.2f} median={candidate_lens['median']}")

# 7. Punctuation distribution
baseline_punct = compute_punctuation_distribution(baseline_lines)
candidate_punct = compute_punctuation_distribution(candidate_lines)
results["punct_baseline"] = baseline_punct
results["punct_candidate"] = candidate_punct
print(f"\n[7] Punctuation distribution (%):")
print(f"    Baseline:    .={baseline_punct['period']:.1f}% ?={baseline_punct['question']:.1f}% !={baseline_punct['exclamation']:.1f}%")
print(f"    Candidate:   .={candidate_punct['period']:.1f}% ?={candidate_punct['question']:.1f}% !={candidate_punct['exclamation']:.1f}%")

# 8. Complex sentence ratio
baseline_complex = compute_complex_ratio(baseline_lines)
candidate_complex = compute_complex_ratio(candidate_lines)
results["complex_baseline"] = baseline_complex
results["complex_candidate"] = candidate_complex
print(f"\n[8] Compound/complex sentence ratio:")
print(f"    Baseline:    {baseline_complex:.1f}%")
print(f"    Candidate:   {candidate_complex:.1f}%")

# 9. Negation ratio
baseline_neg = compute_negation_ratio(baseline_lines)
candidate_neg = compute_negation_ratio(candidate_lines)
results["neg_baseline"] = baseline_neg
results["neg_candidate"] = candidate_neg
print(f"\n[9] Negation sentence ratio:")
print(f"    Baseline:    {baseline_neg:.1f}%")
print(f"    Candidate:   {candidate_neg:.1f}%")

# 10. Template coverage (2-word pattern diversity)
baseline_tmpl = compute_template_coverage(baseline_lines)
candidate_tmpl = compute_template_coverage(candidate_lines)
results["template_baseline"] = baseline_tmpl
results["template_candidate"] = candidate_tmpl
print(f"\n[10] Template pattern coverage (2-word prefix):")
print(f"     Baseline:    {baseline_tmpl['unique_patterns_2word']} unique patterns, top-1={baseline_tmpl['top1_pattern_pct']:.1f}%, top-5={baseline_tmpl['top5_patterns_pct']:.1f}%")
print(f"     Candidate:   {candidate_tmpl['unique_patterns_2word']} unique patterns, top-1={candidate_tmpl['top1_pattern_pct']:.1f}%, top-5={candidate_tmpl['top5_patterns_pct']:.1f}%")

# 11. Cross-set overlap with existing splits
ref_sets = {
    "train": train_lines,
    "val": val_lines,
    "test": test_lines,
}
cross_overlap = compute_cross_set_overlap(candidate_lines, ref_sets)
results["cross_overlap"] = cross_overlap
print(f"\n[11] Cross-set overlap (sentence-level):")
for name, info in cross_overlap.items():
    print(f"     {name}: {info['count']} overlapping sentences ({info['pct_of_candidate']:.2f}% of candidate)")
    if info["examples"]:
        for ex in info["examples"][:3]:
            print(f"       Example: {repr(ex[:60])}")

# 12. Tokenizer coverage
tokenizer_coverage = compute_tokenizer_coverage(candidate_lines)
results["tokenizer_coverage"] = tokenizer_coverage
print(f"\n[12] Tokenizer vocabulary coverage:")
print(f"     Total unique words: {tokenizer_coverage['total_unique_words']}")
print(f"     In vocab:           {tokenizer_coverage['in_vocab']} ({tokenizer_coverage['in_vocab']/max(1,tokenizer_coverage['total_unique_words'])*100:.1f}%)")
print(f"     Missing from vocab: {tokenizer_coverage['missing_from_vocab']}")
if tokenizer_coverage["missing_words"]:
    print(f"     Missing examples:   {tokenizer_coverage['missing_words'][:15]}")

# 13. Sample output
print(f"\n[13] Sample sentences (first 20):")
for i, s in enumerate(candidate_lines[:20], 1):
    print(f"     {i:2d}. {s}")

print(f"\n[14] Sample sentences from each category (5 each):")
# We can't easily categorize without knowing which template generated each sentence,
# but we can show samples by type
types_found = {"declarative": [], "question": [], "exclamation": [], "compound": []}
for s in candidate_lines:
    if "?" in s and types_found["question"].__len__() < 5:
        types_found["question"].append(s)
    elif "!" in s and types_found["exclamation"].__len__() < 5:
        types_found["exclamation"].append(s)
    elif any(c in s for c in [" and ", " so ", " yet "]) and types_found["compound"].__len__() < 5:
        types_found["compound"].append(s)
    elif len(types_found["declarative"]) < 5:
        types_found["declarative"].append(s)

for typ, samples in types_found.items():
    print(f"     {typ}:")
    for s in samples[:5]:
        print(f"       {s}")

# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 60)
print("SUMMARY COMPARISON")
print("=" * 60)

improvements = []
regressions = []
targets = {
    "2gram_rate_candidate": ("2-gram repeat rate", 50.0, "<"),
    "top3_first_candidate": ("Top-3 first word concentration", 40.0, "<"),
    "vocab_util_candidate": ("Vocabulary utilization", 60.0, ">"),
    "herdan_candidate": ("Herdan's C", 0.65, ">"),
    "length_candidate_stdev": ("Sentence length std dev", 2.0, ">"),
    "punct_candidate_question": ("Question mark %", 10.0, ">"),
    "punct_candidate_exclamation": ("Exclamation %", 3.0, ">"),
    "complex_candidate": ("Compound/complex %", 15.0, ">"),
    "template_candidate_top1_pattern_pct": ("Top template pattern %", 40.0, "<"),
}

for key, (label, target_val, direction) in targets.items():
    if key in results:
        actual = results[key]
        if isinstance(actual, dict):
            actual = actual.get("value", 0)
        if direction == "<":
            passed = actual < target_val
            status = "PASS" if passed else "FAIL"
        else:
            passed = actual > target_val
            status = "PASS" if passed else "FAIL"
        sign = "<" if direction == "<" else ">"
        print(f"  [{status}] {label}: {actual:.2f} {sign} {target_val}")
        if passed:
            improvements.append(f"{label}: {actual:.2f} {sign} {target_val}")
        else:
            regressions.append(f"{label}: {actual:.2f} {sign} {target_val} (target: {target_val})")

print(f"\n  Improvements over baseline: {len(improvements)}")
print(f"  Targets not met: {len(regressions)}")

# Save results to JSON for programmatic access
output_json = root / "experiments" / "b4" / "data" / "audit_results.json"
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n  Results saved to: {output_json}")

print("\n" + "=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)
