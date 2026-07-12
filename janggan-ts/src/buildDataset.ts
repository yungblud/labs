// 장단 데이터셋 빌더 — 파이썬 data/build_dataset.py 이식.
// shorthand 문자열(한 글자=한 셀)을 그리드로 펼쳐 data/patterns.json 생성.
// 변형은 손저작(파이썬판과 동일).

import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { LEGEND, GU, type Cell, type Grid, type Pattern } from "./tokens.ts";

// 굿거리 4×3 / 자진모리 4×3 / 세마치 3×3 / 엇모리 3+2+3+2
const JANGDAN: Record<string, string[][]> = {
  gutgeori: [
    ["D.G", "K.R", "D.G", "K.R"],
    ["D.G", "K.R", "D.G", "KR."],
    ["DKG", "K.R", "D.G", "K.R"],
    ["D.G", "K.T", "D.G", "K.R"],
    ["D.G", "K.R", "DKG", "K.R"],
    ["D..", "K.R", "D.G", "K.R"],
    ["D.G", "K.R", "D.T", "K.R"],
    ["DTG", "K.R", "DTG", "K.R"],
    ["D.G", "K.R", "D.G", "KTR"],
  ],
  jajinmori: [
    ["DTK", "T.K", "DTK", "TKT"],
    ["DTK", "TTK", "DTK", "TKT"],
    ["DTK", "T.K", "DTK", "TKK"],
    ["DTG", "T.K", "DTK", "TKT"],
    ["DTK", "T.K", "DGK", "TKT"],
    ["DKK", "T.K", "DTK", "TKT"],
    ["DTK", "TTK", "DTK", "TKG"],
    ["DTK", "T.K", "DTK", "T.T"],
    ["DTK", "TGK", "DTK", "TKT"],
  ],
  semachi: [
    ["D.K", "T.K", "DT."],
    ["D.K", "T.K", "DTG"],
    ["DGK", "T.K", "DT."],
    ["D.K", "TTK", "DT."],
    ["D.K", "T.K", "DTK"],
    ["D.G", "T.K", "DT."],
    ["DKK", "T.K", "DT."],
    ["D.K", "TGK", "DT."],
  ],
  eotmori: [
    ["D.K", "TK", "D.G", "TK"],
    ["D.K", "TK", "D.G", "TT"],
    ["DGK", "TK", "D.G", "TK"],
    ["D.K", "GK", "D.G", "TK"],
    ["D.K", "TK", "DGG", "TK"],
    ["D.K", "TK", "D.K", "TK"],
    ["DKK", "TK", "D.G", "TK"],
    ["D.K", "TK", "D.G", "KT"],
  ],
};

function expand(beatStrs: string[]): Grid {
  return beatStrs.map((bs) =>
    [...bs].map((ch) => {
      if (!(ch in LEGEND)) throw new Error(`unknown shorthand char: ${ch} in ${bs}`);
      return LEGEND[ch] as Cell;
    }),
  );
}

function subdivisions(grid: Grid): number | number[] {
  const lens = grid.map((b) => b.length);
  return new Set(lens).size === 1 ? lens[0] : lens;
}

export function buildPatterns(): { meta: unknown; patterns: Pattern[] } {
  const patterns: Pattern[] = [];
  for (const [name, variants] of Object.entries(JANGDAN)) {
    variants.forEach((beatStrs, i) => {
      const grid = expand(beatStrs);
      patterns.push({
        name,
        variant: i === 0 ? "basic" : `v${i}`,
        beats: grid.length,
        subdivisions: subdivisions(grid),
        grid,
      });
    });
  }
  return {
    meta: {
      gu: GU,
      note: "연구용 양식화 근사치. buildDataset.ts 로 생성. 규칙=../../janggan/data/NOTATION.md.",
    },
    patterns,
  };
}

const HERE = dirname(fileURLToPath(import.meta.url));
export const DATA_PATH = join(HERE, "..", "data", "patterns.json");

if (import.meta.url === `file://${process.argv[1]}`) {
  const data = buildPatterns();
  writeFileSync(DATA_PATH, `${JSON.stringify(data, null, 2)}\n`);
  console.log(`wrote ${data.patterns.length} patterns → ${DATA_PATH}`);
}
