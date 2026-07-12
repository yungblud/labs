// 5단계 — 생성물 구조 유효성 비교 — 파이썬 compare.py 이식.
//   parse / cycle(공통) / shape(v2전용) / novel.

import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { BOS, EOS, PAD, GU_SET, type Cell, type Grid } from "./tokens.ts";
import { ENCODERS, loadPatterns } from "./encode.ts";
import { build } from "./dataset.ts";
import { TinyGPT } from "./model.ts";

const N = 200;
const TEMPS = [1.0, 1.5, 2.0];
const SEED = 7;

const HERE = dirname(fileURLToPath(import.meta.url));
const RUNS = join(HERE, "..", "runs");

// ── 안전 파서 (생성물은 malformed 일 수 있음) ────────────────────────────────
function v1Parse(toks: string[]): Cell[] | null {
  if (toks.length === 0 || toks.length % 2 !== 0) return null;
  const flat: Cell[] = [];
  for (let a = 0; a < toks.length; a += 2) {
    const sym = toks[a];
    const dur = toks[a + 1];
    if (!GU_SET.has(sym) && sym !== "rest") return null;
    if (!/^d\d+$/.test(dur) || Number(dur.slice(1)) < 1) return null;
    flat.push(sym === "rest" ? null : (sym as Cell));
    for (let r = 0; r < Number(dur.slice(1)) - 1; r++) flat.push(null);
  }
  return flat;
}

function v2Parse(toks: string[]): Grid | null {
  if (toks.length === 0 || toks[0][0] !== "|") return null;
  const grid: Grid = [];
  let beat: Cell[] | null = null;
  let i = 0;
  while (i < toks.length) {
    const t = toks[i];
    if (t[0] === "|") {
      if (beat !== null) grid.push(beat);
      if (!/^\d+$/.test(t.slice(1))) return null;
      beat = new Array(Number(t.slice(1))).fill(null);
      i += 1;
    } else if (t[0] === "p") {
      if (!/^\d+$/.test(t.slice(1))) return null;
      const pos = Number(t.slice(1));
      if (beat === null || pos >= beat.length || i + 1 >= toks.length || !GU_SET.has(toks[i + 1])) return null;
      if (beat[pos] !== null) return null;
      beat[pos] = toks[i + 1] as Cell;
      i += 2;
    } else {
      return null;
    }
  }
  if (beat !== null) grid.push(beat);
  return grid;
}

interface Row { temp: number; parse: number; cycle: number; shape: number | null; novel: number }

function measure(scheme: string, validTotals: Set<number>, validShapes: Set<string>) {
  const patterns = loadPatterns();
  const { vocab, ids } = build(patterns, scheme);
  const model = TinyGPT.fromJSON(readFileSync(join(RUNS, `${scheme}.model.json`), "utf-8"));
  const parse = scheme === "v1" ? v1Parse : v2Parse;
  const trainSet = new Set(patterns.map((p) => ENCODERS[scheme](p.grid).join(" ")));

  const rows: Row[] = [];
  let sample = "";
  for (const temp of TEMPS) {
    let ok = 0, cyc = 0, shp = 0, nov = 0;
    for (let n = 0; n < N; n++) {
      const out = model.generate(ids.bos, ids.eos, model.cfg.block, temp, SEED * 1000 + n * 13);
      const toks = out.map((i) => vocab[i]).filter((t) => t !== BOS && t !== EOS && t !== PAD);
      const parsed = parse(toks);
      if (parsed === null) continue;
      ok++;
      const total = scheme === "v1"
        ? (parsed as Cell[]).length
        : (parsed as Grid).reduce((s, b) => s + b.length, 0);
      if (validTotals.has(total)) cyc++;
      if (scheme === "v2" && validShapes.has((parsed as Grid).map((b) => b.length).join(","))) shp++;
      if (!trainSet.has(toks.join(" "))) nov++;
      if (temp === TEMPS[TEMPS.length - 1] && !sample) sample = toks.join(" ");
    }
    rows.push({
      temp,
      parse: +(100 * ok / N).toFixed(1),
      cycle: +(100 * cyc / N).toFixed(1),
      shape: scheme === "v2" ? +(100 * shp / N).toFixed(1) : null,
      novel: +(100 * nov / N).toFixed(1),
    });
  }
  return { rows, sample };
}

const patterns = loadPatterns();
const validTotals = new Set(patterns.map((p) => p.grid.reduce((s, b) => s + b.length, 0)));
const validShapes = new Set(patterns.map((p) => p.grid.map((b) => b.length).join(",")));
console.log(`N=${N}/temp  valid_totals=${[...validTotals].sort()}  valid_shapes=${[...validShapes].sort()}\n`);

const report: Record<string, Row[]> = {};
for (const scheme of ["v1", "v2"]) {
  const { rows, sample } = measure(scheme, validTotals, validShapes);
  report[scheme] = rows;
  console.log(`[${scheme}]`);
  for (const r of rows) {
    const shp = r.shape !== null ? ` shape=${r.shape.toString().padStart(5)}` : "";
    console.log(`  temp ${r.temp}  parse=${r.parse.toString().padStart(5)}%  cycle=${r.cycle.toString().padStart(5)}%${shp}  novel=${r.novel.toString().padStart(5)}%`);
  }
  console.log(`  예시 생성(temp ${TEMPS[TEMPS.length - 1]}): ${sample || "—"}\n`);
}

writeFileSync(join(RUNS, "compare.json"), JSON.stringify(report, null, 2));
console.log("saved → runs/compare.json");
