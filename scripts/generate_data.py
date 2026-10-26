"""
Synthetic training data generator for AronaLM Phase A.

Generates simple English sentences with controlled vocabulary
to validate the training pipeline without external data dependencies.
"""

from pathlib import Path


def generate_synthetic_corpus(
    n_sentences: int = 5000,
    vocab_words: list[str] | None = None,
    seed: int = 42,
) -> str:
    """
    Generate synthetic English-like text corpus.

    Args:
        n_sentences: number of sentences to generate
        vocab_words: custom vocabulary list (default: small fixed set)
        seed: random seed for reproducibility
    """
    import random
    random.seed(seed)

    if vocab_words is None:
        vocab_words = [
            # Nouns
            "the", "cat", "dog", "bird", "fish", "house", "tree", "book", "table", "chair",
            "child", "man", "woman", "friend", "world", "time", "day", "night", "hand", "head",
            # Verbs
            "runs", "jumps", "sits", "likes", "sees", "has", "is", "was", "becomes", "makes",
            "goes", "comes", "takes", "gives", "knows", "wants", "needs", "finds", "puts", "holds",
            # Adjectives
            "big", "small", "old", "new", "good", "bad", "long", "short", "high", "low",
            "dark", "light", "warm", "cold", "bright", "soft", "hard", "quick", "slow", "deep",
            # Adverbs
            "quickly", "slowly", "warmly", "deeply", "brightly", "softly", "gently", "fully",
            # Prepositions / particles
            "on", "in", "under", "near", "behind", "before", "after", "through", "across", "around",
            # Determiners
            "a", "an", "some", "many", "each", "every", "other", "such",
            # Conjunctions
            "and", "but", "or", "yet", "so",
            # Pronouns
            "it", "they", "we", "you", "he", "she",
            # Common modifiers
            "very", "quite", "really", "more", "most", "less",
        ]

    templates = [
        "The {adj} {noun} {verb} {adv}.",
        "A {adj} {noun} {verb} {prep} the {adj} {noun}.",
        "{noun} {verb} the {adj} {noun} {adv}.",
        "The {adj} {noun} {verb} {adj} and {adj}.",
        "I see a {adj} {noun} {verb} {adv}.",
        "{adj} {noun} {verb} {prep} the {adj} {noun}.",
        "The {noun} {verb} {adv} through the {adj} {noun}.",
        "A {noun} {verb} on the {adj} {noun}.",
        "{noun} and {noun} {verb} {prep} the {adj} {noun}.",
        "The {adj} {noun} {verb} {adv} near the {noun}.",
    ]

    sentences = []
    for _ in range(n_sentences):
        template = random.choice(templates)
        try:
            sentence = template.format(
                adj=random.choice([w for w in vocab_words if w in [
                    "big", "small", "old", "new", "good", "bad", "long", "short",
                    "high", "low", "dark", "light", "warm", "cold", "bright",
                    "soft", "hard", "quick", "slow", "deep", "very", "quite",
                ]]),
                noun=random.choice([w for w in vocab_words if w not in [
                    "the", "a", "an", "on", "in", "and", "but", "or", "runs",
                    "jumps", "sits", "likes", "sees", "has", "is", "was",
                ]]),
                verb=random.choice([w for w in vocab_words if w in [
                    "runs", "jumps", "sits", "likes", "sees", "has", "is", "was",
                    "becomes", "makes", "goes", "comes", "takes", "gives",
                    "knows", "wants", "needs", "finds", "puts", "holds",
                ]]),
                adv=random.choice([w for w in vocab_words if w in [
                    "quickly", "slowly", "warmly", "deeply", "brightly",
                    "softly", "gently", "fully", "very", "quite", "really",
                ]]),
                prep=random.choice([w for w in vocab_words if w in [
                    "on", "in", "under", "near", "behind", "before", "after",
                    "through", "across", "around",
                ]]),
            )
        except KeyError:
            # Fallback: generate a simple sentence
            noun1 = random.choice(vocab_words)
            noun2 = random.choice(vocab_words)
            verb = random.choice(["runs", "jumps", "sits", "likes", "sees", "has", "goes"])
            adj = random.choice(["big", "small", "old", "new", "good", "dark", "bright"])
            sentence = f"The {adj} {noun1} {verb} {adj} {noun2}."

        sentences.append(sentence)

    return "\n".join(sentences)


def save_corpus(output_path: Path, n_sentences: int = 5000):
    """Generate and save synthetic corpus to file."""
    corpus = generate_synthetic_corpus(n_sentences=n_sentences)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(corpus)
    return corpus


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw/corpus.txt"))
    parser.add_argument("--n-sentences", type=int, default=5000)
    args = parser.parse_args()

    corpus = save_corpus(args.output, args.n_sentences)
    print(f"Generated {args.n_sentences} sentences -> {args.output}")
    print(f"Token count (whitespace-split): {len(corpus.split())}")
    print(f"Character count: {len(corpus)}")

