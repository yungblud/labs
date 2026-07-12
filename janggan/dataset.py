"""토큰 시퀀스 → 학습용 텐서. v1/v2 공용 (인코더만 갈아끼움). 4·5단계 공용."""

from __future__ import annotations

import torch

from encode import v1_encode, v2_encode

BOS, EOS, PAD = "<bos>", "<eos>", "<pad>"
ENCODERS = {"v1": v1_encode, "v2": v2_encode}


def build(patterns, scheme):
    """→ (vocab, stoi, itos, data[N,maxlen], maxlen, ids dict)."""
    encode = ENCODERS[scheme]
    seqs = [[BOS] + encode(p["grid"]) + [EOS] for p in patterns]

    vocab = [PAD, BOS, EOS] + sorted({t for s in seqs for t in s} - {BOS, EOS})
    stoi = {t: i for i, t in enumerate(vocab)}
    itos = {i: t for t, i in stoi.items()}
    maxlen = max(len(s) for s in seqs)

    data = torch.full((len(seqs), maxlen), stoi[PAD], dtype=torch.long)
    for i, s in enumerate(seqs):
        ids = torch.tensor([stoi[t] for t in s])
        data[i, : len(ids)] = ids

    ids = {"pad": stoi[PAD], "bos": stoi[BOS], "eos": stoi[EOS]}
    return vocab, stoi, itos, data, maxlen, ids
