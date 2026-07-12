// 초소형 GPT (tfjs, from scratch) — 파이썬 model.py 이식.
// 이식 단순화: batch=1 가변길이, single-head, 패딩 없음(스펙의 의도된 차이).

import * as tf from "@tensorflow/tfjs-node";

export interface Config {
  vocab: number;
  block: number;
  d: number;
  layers: number;
}

const STD = 0.02;

interface Layer {
  ln1g: tf.Variable; ln1b: tf.Variable;
  wq: tf.Variable; wk: tf.Variable; wv: tf.Variable; wo: tf.Variable;
  ln2g: tf.Variable; ln2b: tf.Variable;
  w1: tf.Variable; b1: tf.Variable; w2: tf.Variable; b2: tf.Variable;
}

function v(shape: number[], init: "randn" | "zeros" | "ones"): tf.Variable {
  if (init === "zeros") return tf.variable(tf.zeros(shape));
  if (init === "ones") return tf.variable(tf.ones(shape));
  return tf.variable(tf.randomNormal(shape, 0, STD));
}

export class TinyGPT {
  cfg: Config;
  tokEmb: tf.Variable;
  posEmb: tf.Variable;
  blocks: Layer[];
  lnfg: tf.Variable; lnfb: tf.Variable;
  headW: tf.Variable; headB: tf.Variable;

  constructor(cfg: Config) {
    this.cfg = cfg;
    const { vocab, block, d, layers } = cfg;
    this.tokEmb = v([vocab, d], "randn");
    this.posEmb = v([block, d], "randn");
    this.blocks = Array.from({ length: layers }, () => ({
      ln1g: v([d], "ones"), ln1b: v([d], "zeros"),
      wq: v([d, d], "randn"), wk: v([d, d], "randn"), wv: v([d, d], "randn"), wo: v([d, d], "randn"),
      ln2g: v([d], "ones"), ln2b: v([d], "zeros"),
      w1: v([d, 4 * d], "randn"), b1: v([4 * d], "zeros"),
      w2: v([4 * d, d], "randn"), b2: v([d], "zeros"),
    }));
    this.lnfg = v([d], "ones");
    this.lnfb = v([d], "zeros");
    this.headW = v([d, vocab], "randn");
    this.headB = v([vocab], "zeros");
  }

  vars(): tf.Variable[] {
    const out: tf.Variable[] = [this.tokEmb, this.posEmb, this.lnfg, this.lnfb, this.headW, this.headB];
    for (const b of this.blocks) {
      out.push(b.ln1g, b.ln1b, b.wq, b.wk, b.wv, b.wo, b.ln2g, b.ln2b, b.w1, b.b1, b.w2, b.b2);
    }
    return out;
  }

  // logits [T, vocab]
  forward(ids: number[]): tf.Tensor2D {
    return tf.tidy(() => {
      const T = ids.length;
      const d = this.cfg.d;
      const tok = tf.gather(this.tokEmb, tf.tensor1d(ids, "int32")) as tf.Tensor2D;
      const pos = this.posEmb.slice([0, 0], [T, d]) as tf.Tensor2D;
      let x = tok.add(pos) as tf.Tensor2D;

      // causal mask [T,T]: j>i → -1e9
      const mask = tf.linalg
        .bandPart(tf.ones([T, T]), -1, 0)
        .sub(1)
        .mul(1e9) as tf.Tensor2D; // 하삼각=0, 상삼각=-1e9

      for (const b of this.blocks) {
        const h = ln(x, b.ln1g, b.ln1b);
        const q = h.matMul(b.wq) as tf.Tensor2D;
        const k = h.matMul(b.wk) as tf.Tensor2D;
        const val = h.matMul(b.wv) as tf.Tensor2D;
        const scores = q.matMul(k, false, true).div(Math.sqrt(d)).add(mask) as tf.Tensor2D;
        const attn = tf.softmax(scores, -1) as tf.Tensor2D;
        const ctx = attn.matMul(val).matMul(b.wo) as tf.Tensor2D;
        x = x.add(ctx) as tf.Tensor2D;
        const h2 = ln(x, b.ln2g, b.ln2b);
        const mlp = gelu(h2.matMul(b.w1).add(b.b1) as tf.Tensor2D).matMul(b.w2).add(b.b2) as tf.Tensor2D;
        x = x.add(mlp) as tf.Tensor2D;
      }
      const xf = ln(x, this.lnfg, this.lnfb);
      return xf.matMul(this.headW).add(this.headB) as tf.Tensor2D;
    });
  }

  // 다음 토큰 예측 손실 (ids[:-1] → ids[1:])
  loss(ids: number[]): tf.Scalar {
    return tf.tidy(() => {
      const logits = this.forward(ids.slice(0, -1));
      const targets = tf.tensor1d(ids.slice(1), "int32");
      const oneHot = tf.oneHot(targets, this.cfg.vocab) as tf.Tensor2D;
      const logp = tf.logSoftmax(logits, -1) as tf.Tensor2D;
      return oneHot.mul(logp).sum().neg().div(ids.length - 1) as tf.Scalar;
    });
  }

  // <bos> 시작 → EOS 까지 자기회귀 생성 (seed 로 재현)
  generate(bos: number, eos: number, maxNew: number, temperature: number, seed: number): number[] {
    const ids = [bos];
    for (let s = 0; s < maxNew; s++) {
      const nxt = tf.tidy(() => {
        const ctx = ids.slice(-this.cfg.block);
        const logits = this.forward(ctx);
        const last = logits.slice([ctx.length - 1, 0], [1, this.cfg.vocab]).div(temperature) as tf.Tensor2D;
        return tf.multinomial(last, 1, seed + s).dataSync()[0];
      });
      ids.push(nxt);
      if (nxt === eos) break;
    }
    return ids;
  }

  toJSON(): string {
    const dump: Record<string, { shape: number[]; data: number[] }> = {};
    for (const [i, variable] of this.vars().entries()) {
      dump[`w${i}`] = { shape: variable.shape as number[], data: Array.from(variable.dataSync()) };
    }
    return JSON.stringify({ cfg: this.cfg, vars: dump });
  }

  static fromJSON(json: string): TinyGPT {
    const parsed = JSON.parse(json);
    const m = new TinyGPT(parsed.cfg);
    const vars = m.vars();
    Object.values(parsed.vars).forEach((w, i) => {
      const { shape, data } = w as { shape: number[]; data: number[] };
      vars[i].assign(tf.tensor(data, shape));
    });
    return m;
  }
}

// GELU (정확형): 0.5·x·(1 + erf(x/√2))
function gelu(x: tf.Tensor2D): tf.Tensor2D {
  return tf.tidy(
    () => x.mul(0.5).mul(tf.erf(x.div(Math.SQRT2)).add(1)) as tf.Tensor2D,
  );
}

// LayerNorm (마지막 축)
function ln(x: tf.Tensor2D, g: tf.Variable, b: tf.Variable): tf.Tensor2D {
  return tf.tidy(() => {
    const mean = x.mean(-1, true);
    const variance = x.sub(mean).square().mean(-1, true);
    return x.sub(mean).div(variance.add(1e-5).sqrt()).mul(g).add(b) as tf.Tensor2D;
  });
}
