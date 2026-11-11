"""
Tokenizer / model vocabulary compatibility tests (方案 B: 声明修正 + 文档补充).

These tests pin the *contract* that makes the shipped checkpoint safe without
retraining or reshaping it:

    actual vocabulary entries  <=  declared vocab_size  ==  model embedding rows

They also cover vocabulary boundaries, special tokens, model dimensionality and
out-of-range input handling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aronalm.checkpoint import load_checkpoint
from aronalm.model import AronaLM, TransformerConfig
from aronalm.tokenizer import BPETokenizer

VOCAB_PATH = PROJECT_ROOT / "tokenizer" / "vocab.json"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "checkpoint_step200.pt"


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    assert VOCAB_PATH.exists(), f"missing tokenizer artifact: {VOCAB_PATH}"
    return BPETokenizer.load(VOCAB_PATH)


@pytest.fixture(scope="module")
def checkpoint_model():
    assert CHECKPOINT_PATH.exists(), f"missing checkpoint: {CHECKPOINT_PATH}"
    model, _opt, config, _meta = load_checkpoint(CHECKPOINT_PATH, torch.device("cpu"))
    return model, config


class TestDeclaredVersusActualVocabulary:
    """声明 vocab_size 与实际词表条目数之间的关系。"""

    def test_declared_vocab_size_is_1000(self, tokenizer):
        assert tokenizer.vocab_size == 1000

    def test_actual_vocab_size_is_smaller_or_equal(self, tokenizer):
        assert tokenizer._actual_vocab_size == len(tokenizer.vocab)
        assert tokenizer._actual_vocab_size <= tokenizer.vocab_size

    def test_actual_vocab_size_matches_training_corpus_result(self, tokenizer):
        # 301 merges over 256 byte symbols + 1 whitespace symbol -> 558 entries
        assert tokenizer._actual_vocab_size == 558
        assert len(tokenizer.merge_rules) == 302

    def test_vocabulary_ids_are_contiguous_from_zero(self, tokenizer):
        ids = sorted(tokenizer.vocab.values())
        assert ids == list(range(len(ids))), "vocabulary ids must be 0..N-1 without holes"
        assert ids[-1] == tokenizer._actual_vocab_size - 1

    def test_reverse_vocab_is_consistent(self, tokenizer):
        assert len(tokenizer.reverse_vocab) == tokenizer._actual_vocab_size
        for symbol, token_id in tokenizer.vocab.items():
            assert tokenizer.reverse_vocab[token_id] == symbol

    def test_actual_vocab_size_is_read_only(self, tokenizer):
        with pytest.raises(AttributeError):
            tokenizer._actual_vocab_size = 1000  # type: ignore[misc]


class TestEncodingBoundaries:
    """encode() 永不产生超出实际词表或模型维度的 id。"""

    @pytest.mark.parametrize(
        "text",
        [
            "the ",
            "hello world",
            "The quick brown fox jumps over the lazy dog.",
            "".join(chr(c) for c in range(32, 127)),
            "multi\nline\ttabbed text",
            "unicode: 你好，Arona",
        ],
    )
    def test_all_encoded_ids_are_in_actual_vocab_range(self, tokenizer, text):
        ids = tokenizer.encode(text)
        assert ids, "expected at least one token"
        assert all(0 <= token_id < tokenizer._actual_vocab_size for token_id in ids)

    def test_all_encoded_ids_fit_model_embedding_rows(self, tokenizer, checkpoint_model):
        _model, config = checkpoint_model
        ids = tokenizer.encode("the quick brown fox jumps over the lazy dog" * 3)
        assert max(ids) < config.vocab_size
        assert max(ids) < tokenizer._actual_vocab_size

    def test_empty_text_encodes_to_empty_list(self, tokenizer):
        assert tokenizer.encode("") == []

    def test_encode_decode_roundtrip_ascii(self, tokenizer):
        text = "good house. The"
        assert tokenizer.decode(tokenizer.encode(text)) == text


class TestSpecialTokens:
    """特殊 token 语义：pad/unk 是合法词表项且不与越界 id 混淆。"""

    def test_pad_and_unk_ids_inside_vocab(self, tokenizer):
        assert 0 <= tokenizer._pad_id < tokenizer._actual_vocab_size
        assert 0 <= tokenizer._unk_id < tokenizer._actual_vocab_size

    def test_pad_and_unk_are_distinct_ids(self, tokenizer):
        assert tokenizer._pad_id != tokenizer._unk_id

    def test_special_ids_are_byte_symbols(self, tokenizer):
        # 当前实现中 pad/unk 复用字节符号 0x00 / 0x01（byte-level BPE）
        assert tokenizer.reverse_vocab[tokenizer._pad_id] == chr(0)
        assert tokenizer.reverse_vocab[tokenizer._unk_id] == chr(1)

    def test_unknown_byte_falls_back_without_crash(self, tokenizer):
        # 任何字节都能编码（byte-level），不会产生 unk 之外的空洞
        ids = tokenizer.encode("\x00\x01")
        assert all(0 <= token_id < tokenizer._actual_vocab_size for token_id in ids)


class TestModelDimensionConsistency:
    """模型维度与实际词表一致：embedding/output_proj 行数 == 声明 vocab_size。"""

    def test_embedding_rows_match_declared_vocab_size(self, checkpoint_model, tokenizer):
        model, config = checkpoint_model
        assert model.token_embedding.num_embeddings == tokenizer.vocab_size
        assert model.token_embedding.num_embeddings == config.vocab_size

    def test_output_projection_rows_match_embedding_rows(self, checkpoint_model):
        model, _config = checkpoint_model
        assert model.output_proj.out_features == model.token_embedding.num_embeddings

    def test_last_declared_id_still_forwardable(self, checkpoint_model, tokenizer):
        model, config = checkpoint_model
        upper_bound_id = config.vocab_size - 1
        assert upper_bound_id >= tokenizer._actual_vocab_size
        logits = model.forward(torch.tensor([[upper_bound_id]]))
        assert logits.shape == (1, 1, config.vocab_size)

    def test_last_actual_token_is_forwardable(self, checkpoint_model, tokenizer):
        model, config = checkpoint_model
        last_real_id = tokenizer._actual_vocab_size - 1
        logits = model.forward(torch.tensor([[last_real_id]]))
        assert logits.shape == (1, 1, config.vocab_size)


class TestOutOfRangeInput:
    """越界输入必须明确失败，而不是静默产生错误结果。"""

    def test_token_id_equal_to_vocab_size_raises(self, checkpoint_model):
        model, config = checkpoint_model
        with pytest.raises(IndexError):
            model.forward(torch.tensor([[config.vocab_size]]))

    def test_negative_token_id_raises(self, checkpoint_model):
        model, _config = checkpoint_model
        with pytest.raises(IndexError):
            model.forward(torch.tensor([[-1]]))

    def test_decode_of_out_of_range_id_returns_empty_without_crash(self, tokenizer):
        assert tokenizer.decode([tokenizer.vocab_size]) == ""
        assert tokenizer.decode([999]) == ""

    def test_sequence_longer_than_max_seq_len_raises(self, checkpoint_model, tokenizer):
        model, config = checkpoint_model
        too_long = torch.zeros((1, config.max_seq_len + 1), dtype=torch.long)
        with pytest.raises(AssertionError):
            model.forward(too_long)

    def test_generation_cannot_exceed_max_seq_len(self, checkpoint_model, tokenizer):
        model, config = checkpoint_model
        prompt = torch.tensor([tokenizer.encode("the ")[:1]])  # (1, 1)
        # 生成时每步都会 forward 整条序列，因此 prompt + new 不能超过 max_seq_len
        with pytest.raises(AssertionError):
            model.generate(prompt, max_new_tokens=config.max_seq_len + 1, temperature=0.0)


class TestCheckpointCompatibilityAcrossDirectories:
    """所有已交付 checkpoint 目录的最新 checkpoint 均可加载且词表维度自洽。"""

    @pytest.mark.parametrize(
        "checkpoint_dir",
        [
            "checkpoints",
            "checkpoints_v2",
            "checkpoints_v2b_dropout0",
            "checkpoints_v2b_dropout05",
            "checkpoints_v2c_500steps",
            "checkpoints_v2c_dropout015",
        ],
    )
    def test_checkpoint_loads_and_matches_tokenizer(self, tokenizer, checkpoint_dir):
        directory = PROJECT_ROOT / checkpoint_dir
        checkpoints = sorted(
            directory.glob("checkpoint_step*.pt"),
            key=lambda p: int(p.stem.replace("checkpoint_step", "")),
        )
        if not checkpoints:
            pytest.skip(f"no checkpoints in {checkpoint_dir}")
        model, _opt, config, _meta = load_checkpoint(checkpoints[-1], torch.device("cpu"))
        assert isinstance(model, AronaLM)
        assert config.vocab_size == tokenizer.vocab_size
        assert model.token_embedding.num_embeddings == config.vocab_size
        assert tokenizer._actual_vocab_size <= config.vocab_size
