"""초소형 GPT (from scratch). 4·5단계 공용.

장단 토큰 시퀀스를 자기회귀로 다음 토큰 예측 학습한다. 데이터가 작아 작게 유지한다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, h, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(dropout)
        )

    def forward(self, x, attn_mask, key_padding_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, pad_id,
                 d=64, h=4, layers=2, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.pad_id = pad_id
        self.tok = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(block_size, d)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d, h, dropout) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size)

    def forward(self, idx, targets=None):
        _, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos)[None])
        causal = torch.triu(torch.ones(T, T, device=idx.device, dtype=torch.bool), 1)
        key_pad = idx == self.pad_id
        for b in self.blocks:
            x = b(x, causal, key_pad)
        logits = self.head(self.lnf(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                ignore_index=self.pad_id,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new, eos_id, temperature=1.0):
        """<bos> 시작 토큰들 → EOS 나올 때까지 자기회귀 생성."""
        for _ in range(max_new):
            logits, _ = self(idx[:, -self.block_size:])
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
            if nxt.item() == eos_id:
                break
        return idx
