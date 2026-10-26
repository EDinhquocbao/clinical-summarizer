import torch

from clinsum.summarization.data import SummDataset, collate
from clinsum.summarization.model import PointerGenerator, _blocked
from clinsum.summarization.vocab import SPECIALS, Vocab

ROWS = [
    {"id": "a", "source": "patient given warfarin 5 mg for atrial fibrillation .",
     "target": "warfarin 5 mg ."},
    {"id": "b", "source": "chest pain resolved with aspirin .", "target": "aspirin given ."},
]


def _setup(use_copy, use_coverage):
    torch.manual_seed(0)
    # "warfarin" cố ý nằm ngoài từ điển: chỉ sinh được nhờ copy
    vocab = Vocab(SPECIALS + [".", "mg", "5", "aspirin", "given", "patient", "chest", "pain"])
    ds = SummDataset(ROWS, vocab, max_source_len=50, max_target_len=20)
    batch = collate([ds[0], ds[1]], vocab.pad)
    model = PointerGenerator(len(vocab), emb_dim=8, hidden_dim=8, use_copy=use_copy,
                             use_coverage=use_coverage, pad_id=vocab.pad, unk_id=vocab.unk)
    return vocab, batch, model


def test_vocab_ext_ids_for_oov():
    vocab = Vocab(["<pad>", "<unk>", "<s>", "</s>", "aspirin"])
    ids, oovs = vocab.encode_source_ext(["aspirin", "warfarin", "warfarin"])
    assert ids == [4, 5, 5] and oovs == ["warfarin"]
    assert vocab.encode_target_ext(["warfarin", "heparin"], oovs) == [5, vocab.unk]
    assert vocab.decode_ext([4, 5], oovs) == ["aspirin", "warfarin"]


def test_final_distribution_sums_to_one_and_copies_oov():
    vocab, batch, model = _setup(use_copy=True, use_coverage=False)
    assert batch["max_oovs"] > 0
    enc_out, enc_feat, (h, _) = model.encode(batch["src"], batch["src_len"])
    mask = batch["src"] != vocab.pad
    cov = torch.zeros(batch["src"].shape)
    attn, ctx = model.attend(enc_out, enc_feat, mask, h[0], cov)
    x = model.embedding(torch.full((2,), vocab.bos))
    p, p_gen = model.final_dist(h[0], ctx, x, attn, batch["src_ext"], batch["max_oovs"])
    assert torch.allclose(p.sum(-1), torch.ones(2), atol=1e-5)
    assert p.shape[-1] == len(vocab) + batch["max_oovs"]
    assert p[:, len(vocab):].sum() > 0  # từ OOV nhận được xác suất nhờ copy
    assert ((p_gen > 0) & (p_gen < 1)).all()


def test_target_nll_matches_full_distribution():
    for use_copy in (False, True):
        vocab, batch, model = _setup(use_copy=use_copy, use_coverage=False)
        enc_out, enc_feat, (h, _) = model.encode(batch["src"], batch["src_len"])
        mask = batch["src"] != vocab.pad
        attn, ctx = model.attend(enc_out, enc_feat, mask, h[0], torch.zeros(batch["src"].shape))
        s, ctx, attn = h[0].unsqueeze(1), ctx.unsqueeze(1), attn.unsqueeze(1)  # T = 1
        x = model.embedding(torch.full((2, 1), vocab.bos))
        target = (batch["dec_out"] if use_copy else batch["dec_out_plain"])[:, :1]
        p, _ = model.final_dist(s, ctx, x, attn, batch["src_ext"], batch["max_oovs"])
        expected = -torch.log(p.gather(-1, target.unsqueeze(-1)).squeeze(-1))
        nll, _ = model.target_nll(s, ctx, x, attn, batch["src_ext"], target)
        assert torch.allclose(nll, expected, atol=1e-5)


def test_forward_losses_and_backward():
    for use_copy, use_cov in ((False, False), (True, False), (True, True)):
        _, batch, model = _setup(use_copy, use_cov)
        out = model(batch)
        assert torch.isfinite(out["loss"])
        assert ("coverage" in out) == use_cov
        out["loss"].backward()


def test_overfit_tiny_batch_then_beam_search_copies_oov():
    vocab, batch, model = _setup(use_copy=True, use_coverage=False)
    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(150):
        opt.zero_grad()
        model(batch)["loss"].backward()
        opt.step()
    ex = 0
    n_oov = len(batch["oovs"][ex])
    src_len = int(batch["src_len"][ex])
    ids, p_gen = model.beam_search(batch["src"][ex:ex + 1, :src_len], batch["src_ext"][ex:ex + 1, :src_len],
                                   n_oov, vocab.bos, vocab.eos, beam_size=2, max_len=10, min_len=1)
    words = vocab.decode_ext(ids, batch["oovs"][ex])
    assert "warfarin" in batch["oovs"][ex]
    assert words == "warfarin 5 mg .".split()
    assert p_gen is not None


def test_ngram_blocking():
    assert _blocked([1, 2, 3, 1, 2], 3) == [3]
