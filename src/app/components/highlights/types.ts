export type HighlightColor =
  | "yellow"
  | "green"
  | "blue"
  | "pink"
  | "purple";

export interface Highlight {
  id: string;
  slug: string;
  text: string;
  note: string;
  color: HighlightColor;
  start: number;
  end: number;
  createdAt: number;
  updatedAt: number;
}
