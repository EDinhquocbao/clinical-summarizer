"""Seq2seq + attention, Pointer-Generator và coverage (See, Liu, Manning, ACL 2017).

Một lớp cho cả ba mô hình so sánh:
- use_copy=False, use_coverage=False : Seq2seq BiLSTM + attention Bahdanau (đối sánh)
- use_copy=True,  use_coverage=False : Pointer-Generator
- use_copy=True,  use_coverage=True  : Pointer-Generator + coverage (phương pháp chính)

Công thức (ký hiệu theo bài báo):
  e_i^t   = v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)      a^t = softmax(e^t)
  h*_t    = sum_i a_i^t h_i                                        (context vector)
  P_vocab = softmax(V'(V[s_t; h*_t] + b) + b')
  p_gen   = sigmoid(w_h*^T h*_t + w_s^T s_t + w_x^T x_t + b_ptr)
  P(w)    = p_gen P_vocab(w) + (1 - p_gen) sum_{i: w_i = w} a_i^t
  covloss = sum_i min(a_i^t, c_i^t),   c^t = sum_{t'<t} a^{t'}

Khác bài báo một điểm để huấn luyện nhanh hơn: decoder không dùng "input feeding"
(không nối context vector vào đầu vào LSTM), nên LSTM decoder chạy một lần cho cả chuỗi.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

EPS = 1e-12


class PointerGenerator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        hidden_dim: int = 256,
        use_copy: bool = True,
        use_coverage: bool = False,
        pad_id: int = 0,
        unk_id: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size, self.hidden_dim = vocab_size, hidden_dim
        self.use_copy, self.use_coverage = use_copy, use_coverage
        self.pad_id, self.unk_id = pad_id, unk_id
        H = hidden_dim

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.encoder = nn.LSTM(emb_dim, H, batch_first=True, bidirectional=True)
        self.reduce_h = nn.Linear(2 * H, H)
        self.reduce_c = nn.Linear(2 * H, H)
        self.decoder = nn.LSTM(emb_dim, H, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.W_h = nn.Linear(2 * H, 2 * H, bias=False)
        self.W_s = nn.Linear(H, 2 * H)
        self.w_c = nn.Linear(1, 2 * H, bias=False)
        self.v = nn.Linear(2 * H, 1, bias=False)

        self.out1 = nn.Linear(3 * H, H)
        self.out2 = nn.Linear(H, vocab_size)
        self.p_gen_linear = nn.Linear(2 * H + H + emb_dim, 1)

    # ------------------------------------------------------------------ encoder
    def encode(self, src: torch.Tensor, src_len: torch.Tensor):
        emb = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, src_len.cpu(), batch_first=True, enforce_sorted=False
        )
        out, (h, c) = self.encoder(packed)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=src.size(1))
        h = F.relu(self.reduce_h(torch.cat([h[0], h[1]], dim=-1))).unsqueeze(0)
        c = F.relu(self.reduce_c(torch.cat([c[0], c[1]], dim=-1))).unsqueeze(0)
        return enc_out, self.W_h(enc_out), (h, c)

    # ---------------------------------------------------------------- attention
    def attend(self, enc_out, enc_feat, src_mask, s_t, coverage):
        """s_t: (B, H). Trả về attention (B, S) và context (B, 2H)."""
        feat = enc_feat + self.W_s(s_t).unsqueeze(1)
        if self.use_coverage:
            feat = feat + self.w_c(coverage.unsqueeze(-1))
        scores = self.v(torch.tanh(feat)).squeeze(-1)
        scores = scores.masked_fill(~src_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.bmm(attn.unsqueeze(1), enc_out).squeeze(1)
        return attn, ctx

    def final_dist(self, s, ctx, x_emb, attn, src_ext, max_oovs):
        """Phân phối xác suất trên từ điển (mở rộng nếu có copy). Hỗ trợ cả (B, D) và (B, T, D)."""
        logits = self.out2(self.out1(torch.cat([s, ctx], dim=-1)))
        p_vocab = torch.softmax(logits, dim=-1)
        if not self.use_copy:
            return p_vocab, None
        p_gen = torch.sigmoid(self.p_gen_linear(torch.cat([ctx, s, x_emb], dim=-1)))
        p = p_gen * p_vocab
        if max_oovs > 0:
            p = torch.cat([p, p.new_zeros(*p.shape[:-1], max_oovs)], dim=-1)
        index = src_ext if attn.dim() == 2 else src_ext.unsqueeze(1).expand_as(attn)
        p = p.scatter_add(-1, index, (1 - p_gen) * attn)
        return p, p_gen

    def target_nll(self, s, ctx, x_emb, attn, src_ext, target):
        """-log P(từ đích) cho (B, T), bằng final_dist nhưng không dựng tensor (B, T, V + OOV)
        để tiết kiệm VRAM: P(y) = p_gen * P_vocab(y) + (1 - p_gen) * sum_{i: w_i = y} a_i."""
        logits = self.out2(self.out1(torch.cat([s, ctx], dim=-1)))
        in_vocab = target < self.vocab_size
        y = target.masked_fill(~in_vocab, self.unk_id)
        nll_vocab = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="none").view_as(y)
        if not self.use_copy:
            return nll_vocab, None
        p_gen = torch.sigmoid(self.p_gen_linear(torch.cat([ctx, s, x_emb], dim=-1))).squeeze(-1)
        p_copy = (attn * (src_ext.unsqueeze(1) == target.unsqueeze(-1))).sum(-1)
        p = p_gen * torch.exp(-nll_vocab) * in_vocab + (1 - p_gen) * p_copy
        return -torch.log(p + EPS), p_gen.unsqueeze(-1)

    # ------------------------------------------------------------------ training
    def forward(self, batch: Dict, cov_weight: float = 1.0) -> Dict[str, torch.Tensor]:
        src, src_len = batch["src"], batch["src_len"]
        src_mask = src != self.pad_id
        enc_out, enc_feat, state = self.encode(src, src_len)

        dec_in = batch["dec_in"]
        x_emb = self.dropout(self.embedding(dec_in))
        dec_out, _ = self.decoder(x_emb, state)  # (B, T, H)
        B, T, _ = dec_out.shape

        coverage = enc_out.new_zeros(B, src.size(1))
        attns, ctxs, cov_losses = [], [], []
        for t in range(T):
            # checkpoint: không giữ tensor (B, S, 2H) của từng bước, tính lại khi backward -> đỡ VRAM
            attn, ctx = checkpoint(self.attend, enc_out, enc_feat, src_mask, dec_out[:, t], coverage,
                                   use_reentrant=False)
            if self.use_coverage:
                cov_losses.append(torch.minimum(attn, coverage).sum(-1))
                coverage = coverage + attn
            attns.append(attn)
            ctxs.append(ctx)
        attn_all, ctx_all = torch.stack(attns, 1), torch.stack(ctxs, 1)

        target = batch["dec_out"] if self.use_copy else batch["dec_out_plain"]
        tgt_mask = (target != self.pad_id).float()
        nll, p_gen = self.target_nll(dec_out, ctx_all, x_emb, attn_all, batch["src_ext"], target)
        n_tok = tgt_mask.sum(1).clamp(min=1)
        nll_loss = ((nll * tgt_mask).sum(1) / n_tok).mean()
        out = {"nll": nll_loss, "loss": nll_loss}
        if self.use_coverage:
            cov = ((torch.stack(cov_losses, 1) * tgt_mask).sum(1) / n_tok).mean()
            out["coverage"] = cov
            out["loss"] = nll_loss + cov_weight * cov
        if p_gen is not None:
            out["p_gen"] = (p_gen.squeeze(-1) * tgt_mask).sum() / tgt_mask.sum()
        return out

    # ----------------------------------------------------------------- inference
    @torch.no_grad()
    def beam_search(
        self,
        src: torch.Tensor,
        src_ext: torch.Tensor,
        n_oovs: int,
        bos_id: int,
        eos_id: int,
        beam_size: int = 4,
        max_len: int = 120,
        min_len: int = 20,
        block_ngram: int = 0,
    ) -> Tuple[List[int], Optional[float]]:
        """Beam search cho MỘT văn bản (src: (1, S)). Trả về chỉ số từ (từ điển mở rộng) và p_gen trung bình."""
        src_len = torch.tensor([src.size(1)])
        enc_out, enc_feat, (h, c) = self.encode(src, src_len)
        S = src.size(1)
        src_mask = torch.ones(1, S, dtype=torch.bool, device=src.device)

        beams = [([bos_id], 0.0, h, c, enc_out.new_zeros(1, S), [])]
        finished = []
        for step in range(max_len):
            k = len(beams)
            last = torch.tensor([b[0][-1] for b in beams], device=src.device)
            last = last.masked_fill(last >= self.vocab_size, self.unk_id)
            x_emb = self.embedding(last).unsqueeze(1)
            h0 = torch.cat([b[2] for b in beams], 1)
            c0 = torch.cat([b[3] for b in beams], 1)
            cov = torch.cat([b[4] for b in beams], 0)
            dec, (h1, c1) = self.decoder(x_emb, (h0, c0))
            s_t = dec.squeeze(1)
            attn, ctx = self.attend(
                enc_out.expand(k, -1, -1), enc_feat.expand(k, -1, -1), src_mask.expand(k, -1), s_t, cov
            )
            p, p_gen = self.final_dist(s_t, ctx, x_emb.squeeze(1), attn, src_ext.expand(k, -1), n_oovs)
            logp = torch.log(p + EPS)
            if step < min_len:
                logp[:, eos_id] = -1e9
            logp[:, self.unk_id] -= 1e9  # không sinh <unk>
            if block_ngram:
                for bi, b in enumerate(beams):
                    for w in _blocked(b[0], block_ngram):
                        logp[bi, w] = -1e9
            top_lp, top_ix = logp.topk(beam_size, dim=-1)

            cands = []
            for bi, b in enumerate(beams):
                for lp, ix in zip(top_lp[bi].tolist(), top_ix[bi].tolist()):
                    pg = b[5] + ([p_gen[bi].item()] if p_gen is not None else [])
                    cands.append((b[0] + [ix], b[1] + lp, h1[:, bi:bi + 1], c1[:, bi:bi + 1],
                                  cov[bi:bi + 1] + attn[bi:bi + 1], pg))
            cands.sort(key=lambda x: x[1] / (len(x[0]) - 1), reverse=True)
            # chỉ beam_size ứng viên tốt nhất được giữ; ứng viên kết thúc bằng </s> cũng chiếm một suất
            beams = []
            for cand in cands[:beam_size]:
                (finished if cand[0][-1] == eos_id else beams).append(cand)
            if len(finished) >= beam_size or not beams:
                break
        pool = finished or beams
        best = max(pool, key=lambda x: x[1] / (len(x[0]) - 1))
        ids = [i for i in best[0][1:] if i != eos_id]
        return ids, (sum(best[5]) / len(best[5]) if best[5] else None)


def _blocked(seq: List[int], n: int) -> List[int]:
    """Các từ mà nếu sinh tiếp sẽ lặp lại một n-gram đã có."""
    if len(seq) < n:
        return []
    prefix = tuple(seq[-(n - 1):])
    return [seq[i + n - 1] for i in range(len(seq) - n + 1) if tuple(seq[i:i + n - 1]) == prefix]
