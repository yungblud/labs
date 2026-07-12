// 토큰 시퀀스 → id 배열 + vocab — 파이썬 dataset.py 이식 (패딩 없음).

import { BOS, EOS, PAD, type Pattern } from "./tokens.ts";
import { ENCODERS } from "./encode.ts";

export interface Built {
  vocab: string[];
  stoi: Map<string, number>;
  seqs: number[][]; // <bos> … <eos> id 시퀀스
  block: number;
  ids: { pad: number; bos: number; eos: number };
}

export function build(patterns: Pattern[], scheme: string): Built {
  const encode = ENCODERS[scheme];
  const tokenSeqs = patterns.map((p) => [BOS, ...encode(p.grid), EOS]);

  const rest = new Set<string>();
  for (const s of tokenSeqs) for (const t of s) if (t !== BOS && t !== EOS) rest.add(t);
  const vocab = [PAD, BOS, EOS, ...[...rest].sort()];
  const stoi = new Map(vocab.map((t, i) => [t, i]));

  const seqs = tokenSeqs.map((s) => s.map((t) => stoi.get(t) as number));
  const block = Math.max(...seqs.map((s) => s.length));
  return {
    vocab,
    stoi,
    seqs,
    block,
    ids: { pad: stoi.get(PAD) as number, bos: stoi.get(BOS) as number, eos: stoi.get(EOS) as number },
  };
}
