// 공용 상수 — 파이썬 encode.py / dataset.py 대응.

export type Gu = "deong" | "kung" | "deok" | "gideok" | "roll";
export type Cell = Gu | null;
export type Grid = Cell[][];

export const GU: Gu[] = ["deong", "kung", "deok", "gideok", "roll"];
export const GU_SET = new Set<string>(GU);

export const BOS = "<bos>";
export const EOS = "<eos>";
export const PAD = "<pad>";

// shorthand 글자 → 구음 (null 은 빈 셀)
export const LEGEND: Record<string, Cell> = {
  D: "deong",
  K: "kung",
  T: "deok",
  G: "gideok",
  R: "roll",
  ".": null,
};

export interface Pattern {
  name: string;
  variant: string;
  beats: number;
  subdivisions: number | number[];
  grid: Grid;
}
