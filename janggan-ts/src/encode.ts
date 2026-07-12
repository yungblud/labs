// v1/v2 인코더·디코더 — 파이썬 encode.py 이식.
//   v1 (flat/duration): `deong d2 gideok d1 …`  박 경계 없음
//   v2 (positional):    `|3 p0 deong p2 gideok …`  칸 명시
// 둘 다 그리드 왕복. v1 은 박 길이(beatLens) 외부 주입 필요.

import { readFileSync } from "node:fs";
import { GU_SET, type Cell, type Grid, type Pattern } from "./tokens.ts";
import { DATA_PATH } from "./buildDataset.ts";

// ── 공통 ──────────────────────────────────────────────────────────────────
export function flatten(grid: Grid): { flat: Cell[]; beatLens: number[] } {
  const flat: Cell[] = [];
  const beatLens: number[] = [];
  for (const beat of grid) {
    beatLens.push(beat.length);
    for (const c of beat) flat.push(c);
  }
  return { flat, beatLens };
}

export function reshape(flat: Cell[], beatLens: number[]): Grid {
  const out: Grid = [];
  let i = 0;
  for (const len of beatLens) {
    out.push(flat.slice(i, i + len));
    i += len;
  }
  return out;
}

// ── v1: flat / duration ─────────────────────────────────────────────────────
export function v1Encode(grid: Grid): string[] {
  const { flat } = flatten(grid);
  const n = flat.length;
  const strokeIdx = flat.flatMap((c, k) => (c !== null ? [k] : []));
  const toks: string[] = [];
  if (strokeIdx.length === 0) return ["rest", `d${n}`];
  if (strokeIdx[0] > 0) toks.push("rest", `d${strokeIdx[0]}`);
  strokeIdx.forEach((k, j) => {
    const end = j + 1 < strokeIdx.length ? strokeIdx[j + 1] : n;
    toks.push(flat[k] as string, `d${end - k}`);
  });
  return toks;
}

export function v1Decode(toks: string[]): Cell[] {
  const flat: Cell[] = [];
  for (let a = 0; a < toks.length; a += 2) {
    const sym = toks[a];
    const dur = Number(toks[a + 1].slice(1));
    flat.push(sym === "rest" ? null : (sym as Cell));
    for (let r = 0; r < dur - 1; r++) flat.push(null);
  }
  return flat;
}

// ── v2: positional / 정간보식 ────────────────────────────────────────────────
export function v2Encode(grid: Grid): string[] {
  const toks: string[] = [];
  for (const beat of grid) {
    toks.push(`|${beat.length}`);
    beat.forEach((c, pos) => {
      if (c !== null) toks.push(`p${pos}`, c);
    });
  }
  return toks;
}

export function v2Decode(toks: string[]): Grid {
  const grid: Grid = [];
  let beat: Cell[] | null = null;
  let i = 0;
  while (i < toks.length) {
    const t = toks[i];
    if (t[0] === "|") {
      if (beat !== null) grid.push(beat);
      beat = new Array(Number(t.slice(1))).fill(null);
      i += 1;
    } else if (t[0] === "p") {
      (beat as Cell[])[Number(t.slice(1))] = toks[i + 1] as Cell;
      i += 2;
    } else {
      i += 1;
    }
  }
  if (beat !== null) grid.push(beat);
  return grid;
}

export const ENCODERS: Record<string, (g: Grid) => string[]> = {
  v1: v1Encode,
  v2: v2Encode,
};

// ── 로드·검증 ────────────────────────────────────────────────────────────────
export function loadPatterns(): Pattern[] {
  return JSON.parse(readFileSync(DATA_PATH, "utf-8")).patterns as Pattern[];
}

function eqGrid(a: Grid, b: Grid): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const patterns = loadPatterns();
  let fails = 0;
  for (const p of patterns) {
    const { beatLens } = flatten(p.grid);
    if (!eqGrid(reshape(v1Decode(v1Encode(p.grid)), beatLens), p.grid)) fails++;
    if (!eqGrid(v2Decode(v2Encode(p.grid)), p.grid)) fails++;
  }
  console.log(`round-trip: ${patterns.length} patterns × 2 인코딩 → 실패 ${fails}`);

  const vocab = (enc: (g: Grid) => string[]) =>
    new Set(patterns.flatMap((p) => enc(p.grid)));
  const v1v = vocab(v1Encode);
  const v2v = vocab(v2Encode);
  const dur = [...v1v].filter((t) => /^d\d+$/.test(t)).sort();
  console.log(`\nvocab  v1(flat/duration) = ${v1v.size}   v2(positional) = ${v2v.size}`);
  console.log(`  v1 길이 토큰: ${JSON.stringify(dur)}`);
  console.log(`  v2 어휘     : ${JSON.stringify([...v2v].sort())}`);

  const e = patterns.find((p) => p.name === "eotmori" && p.variant === "basic") as Pattern;
  console.log(`\n엇모리 basic (비대칭 3+2+3+2):`);
  console.log(`  v1: ${v1Encode(e.grid).join(" ")}`);
  console.log(`  v2: ${v2Encode(e.grid).join(" ")}`);
}
