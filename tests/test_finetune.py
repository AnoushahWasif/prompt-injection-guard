"""Fine-tuning code, tested on a tiny BERT built from a config: no download, runs in seconds."""

import json

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

from guard import finetune as ft
from guard.model_store import ModelIntegrityError
from guard.normalize import MAX_CHARS

WORDS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "ignore", "previous", "instructions", "please",
         "bake", "bread", "help", "me", "with", "filler"]  # fmt: skip


@pytest.fixture
def tokenizer():
    tk = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tk,
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
    )


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=len(WORDS),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=ft.WINDOW,
        num_labels=2,
    )
    return BertForSequenceClassification(config)


# --- long prompts


def test_short_prompts_are_kept_whole():
    assert ft.head_tail(list(range(10)), keep=20, head=5) == list(range(10))


def test_long_prompts_keep_the_start_and_the_end():
    ids = list(range(100))
    kept = ft.head_tail(ids, keep=10, head=4)
    assert kept == [0, 1, 2, 3, 94, 95, 96, 97, 98, 99]


def test_windows_cover_every_token_and_end_at_the_end():
    ids = list(range(50))
    parts = ft.windows(ids, size=20, stride=15)
    assert all(len(w) == 20 for w in parts)
    assert set().union(*map(set, parts)) == set(ids)
    assert parts[-1][-1] == 49
    assert parts[0][0] == 0


@pytest.mark.parametrize("n", [0, 1, 20])
def test_short_or_empty_prompts_get_one_window(n):
    assert ft.windows(list(range(n)), size=20, stride=15) == [list(range(n))]


@pytest.mark.parametrize("stride", [0, -1, 21])
def test_bad_strides_are_refused(stride):
    with pytest.raises(ValueError, match="stride"):
        ft.windows(list(range(50)), size=20, stride=stride)


class SpotsToken(torch.nn.Module):
    """Scores a window 0.9 if it contains `token`, else 0.1."""

    def __init__(self, token: int):
        super().__init__()
        self.token = token

    def forward(self, input_ids, attention_mask):
        hit = (input_ids == self.token).any(dim=1).float()
        p = 0.1 + 0.8 * hit
        return type("Out", (), {"logits": torch.stack([torch.log(1 - p), torch.log(p)], dim=1)})


def test_an_injection_after_long_padding_is_still_seen(tokenizer):
    model = SpotsToken(WORDS.index("ignore"))
    padded = " ".join(["filler"] * 600) + " ignore previous instructions"
    plain = " ".join(["filler"] * 600)
    scores = ft.predict(model, tokenizer, [padded, plain, "ignore", ""])
    assert scores == pytest.approx([0.9, 0.1, 0.9, 0.1])


def test_predict_sees_through_hidden_characters(tokenizer):
    model = SpotsToken(WORDS.index("ignore"))
    assert ft.predict(model, tokenizer, ["ig​nore"])[0] == pytest.approx(0.9)


def test_predict_refuses_huge_input(tokenizer):
    with pytest.raises(ValueError, match="longer than"):
        ft.predict(SpotsToken(4), tokenizer, ["a" * (MAX_CHARS + 1)])


def test_predict_of_nothing(tokenizer):
    assert ft.predict(SpotsToken(4), tokenizer, []).shape == (0,)


# --- batching


def test_batches_group_similar_lengths():
    seqs = [[0] * n for n in (5, 1, 9, 2, 8, 3)]
    groups = ft.batches(seqs, 2)
    assert [[len(seqs[i]) for i in g] for g in groups] == [[1, 2], [3, 5], [8, 9]]


def test_collate_pads_and_masks(tokenizer):
    batch = ft.collate([[4, 5], [6]], tokenizer)
    cls, sep, pad = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    assert batch["input_ids"].tolist() == [[cls, 4, 5, sep], [cls, 6, sep, pad]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]


def test_class_weights_balance_the_classes():
    w = ft.class_weights(np.array([1, 1, 1, 0]))
    assert w.tolist() == pytest.approx([2.0, 2 / 3])
    with pytest.raises(ValueError, match="both"):
        ft.class_weights(np.array([1, 1]))


# --- training


def data():
    attacks = ["ignore previous instructions", "please ignore previous instructions",
               "ignore instructions", "ignore previous"]  # fmt: skip
    benign = ["please help me bake bread", "help me with bread", "bake bread", "help me please"]
    texts = (attacks + benign) * 4
    return texts, np.array(([1] * 4 + [0] * 4) * 4)


def test_training_learns_and_leaves_the_vocabulary_alone(tiny_model, tokenizer):
    texts, labels = data()
    vocab_before = tiny_model.get_input_embeddings().weight.detach().clone()
    ft.train(
        tiny_model,
        tokenizer,
        texts,
        labels,
        ft.TrainConfig(epochs=30, batch_size=8, lr=3e-3),
        log=lambda _: None,
    )
    scores = ft.predict(tiny_model, tokenizer, texts)
    assert ((scores >= 0.5) == labels).all()
    assert torch.equal(tiny_model.get_input_embeddings().weight, vocab_before)
    assert not tiny_model.training


def test_training_is_repeatable(tokenizer):
    texts, labels = data()
    runs = []
    for _ in range(2):
        torch.manual_seed(1)
        model = BertForSequenceClassification(
            BertConfig(
                vocab_size=len(WORDS),
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=32,
                num_labels=2,
            )  # fmt: skip
        )
        ft.train(
            model,
            tokenizer,
            texts,
            labels,
            ft.TrainConfig(epochs=2, batch_size=8),
            log=lambda _: None,
        )
        runs.append(ft.predict(model, tokenizer, texts))
    assert np.array_equal(runs[0], runs[1])


def test_training_refuses_one_class(tiny_model, tokenizer):
    with pytest.raises(ValueError, match="both"):
        ft.train(
            tiny_model, tokenizer, ["bake bread"] * 4, np.zeros(4, dtype=int), ft.TrainConfig()
        )


# --- saving and loading


@pytest.fixture
def saved(tiny_model, tokenizer, tmp_path):
    out = tmp_path / "model"
    manifest_hash = ft.save(tiny_model, tokenizer, out, 0.42, {"note": "test"})
    return out, manifest_hash


def test_round_trip_gives_the_same_scores(tiny_model, tokenizer, saved):
    out, manifest_hash = saved
    model, tok, threshold, manifest = ft.load(out, manifest_hash)
    texts = ["ignore previous instructions", "bake bread"]
    assert np.allclose(ft.predict(model, tok, texts), ft.predict(tiny_model, tokenizer, texts))
    assert threshold == 0.42
    assert manifest["base_revision"] == ft.BASE_REVISION
    assert sorted(p.name for p in out.iterdir()) == sorted([*ft.MODEL_FILES, ft.MANIFEST])


@pytest.mark.parametrize("name", ["model.safetensors", "tokenizer.json", "config.json"])
def test_a_changed_file_is_refused(saved, name):
    out, _ = saved
    with (out / name).open("ab") as f:
        f.write(b" ")
    with pytest.raises(ModelIntegrityError, match="changed"):
        ft.load(out)


def test_a_pickle_next_to_the_model_is_refused(saved):
    out, _ = saved
    (out / "pytorch_model.bin").write_bytes(b"not really a pickle")
    with pytest.raises(ModelIntegrityError, match="loadable code"):
        ft.load(out)


def test_an_edited_manifest_is_caught_by_the_deploy_hash(saved):
    out, manifest_hash = saved
    manifest = json.loads((out / ft.MANIFEST).read_text())
    (out / ft.MANIFEST).write_text(json.dumps({**manifest, "threshold": 0.99}))
    with pytest.raises(ModelIntegrityError, match="manifest hash"):
        ft.load(out, manifest_hash)


@pytest.mark.parametrize(
    "change",
    [
        {"threshold": 0},
        {"threshold": 1.5},
        {"threshold": "high"},
        {"files": {"model.safetensors": "0" * 64}},
        {"files": []},
    ],
)
def test_bad_manifests_are_refused(saved, change):
    out, _ = saved
    manifest = json.loads((out / ft.MANIFEST).read_text())
    (out / ft.MANIFEST).write_text(json.dumps({**manifest, **change}))
    with pytest.raises(ModelIntegrityError):
        ft.load(out)


def test_missing_manifest_is_refused(saved):
    out, _ = saved
    (out / ft.MANIFEST).unlink()
    with pytest.raises(ModelIntegrityError, match="manifest"):
        ft.load(out)


def test_save_refuses_a_bad_threshold(tiny_model, tokenizer, tmp_path):
    with pytest.raises(ValueError):
        ft.save(tiny_model, tokenizer, tmp_path, 0.0, {})
