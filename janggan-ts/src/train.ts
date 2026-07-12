// 4단계 — v1/v2 동일 조건 학습 — 파이썬 train.py 이식.
// batch=1 가변길이(패딩 없음), CPU. 변수는 인코딩 하나뿐.

import { mkdirSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import * as tf from "@tensorflow/tfjs-node";
import { loadPatterns } from "./encode.ts";
import { build } from "./dataset.ts";
import { TinyGPT } from "./model.ts";

const D = 32;
const LAYERS = 2;
const EPOCHS = 60; // 34 seq × 60 ≈ 2040 업데이트
const LR = 3e-3;
const SEED = 1337;

const HERE = dirname(fileURLToPath(import.meta.url));
const RUNS = join(HERE, "..", "runs");

// 결정적 셔플 (seed 기반 LCG) — Math.random 회피
function shuffled<T>(arr: T[], seed: number): T[] {
  const a = [...arr];
  let s = seed >>> 0;
  const rand = () => ((s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export function trainScheme(scheme: string): { vocab: number; finalLoss: number } {
  tf.setBackend("tensorflow");
  const patterns = loadPatterns();
  const { vocab, seqs, block, ids } = build(patterns, scheme);
  const model = new TinyGPT({ vocab: vocab.length, block, d: D, layers: LAYERS });
  const opt = tf.train.adam(LR);

  const history: { step: number; loss: number }[] = [];
  let step = 0;
  let last = 0;
  for (let e = 0; e < EPOCHS; e++) {
    for (const seq of shuffled(seqs, SEED + e)) {
      const cost = opt.minimize(() => model.loss(seq), true, model.vars()) as tf.Scalar;
      last = cost.dataSync()[0];
      cost.dispose();
      step++;
      if (step % 200 === 0 || step === 1) {
        history.push({ step, loss: Number(last.toFixed(4)) });
        console.log(`  [${scheme}] step ${step}  loss ${last.toFixed(4)}`);
      }
    }
  }

  mkdirSync(RUNS, { recursive: true });
  writeFileSync(join(RUNS, `${scheme}.model.json`), model.toJSON());
  writeFileSync(join(RUNS, `${scheme}.loss.json`), JSON.stringify(history, null, 2));
  opt.dispose();
  return { vocab: vocab.length, finalLoss: Number(last.toFixed(4)) };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  console.log(`backend=${tf.getBackend()}  epochs=${EPOCHS} seed=${SEED}\n`);
  const results: Record<string, unknown> = {};
  for (const scheme of ["v1", "v2"]) {
    const r = trainScheme(scheme);
    results[scheme] = r;
    console.log(`  → ${scheme}: vocab=${r.vocab} final_loss=${r.finalLoss}\n`);
  }
  console.log("done:", JSON.stringify(results));
}
