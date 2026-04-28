declare module "stopword" {
  export const eng: string[];
  export const vie: string[];
  export function removeStopwords(tokens: string[], list?: string[]): string[];
}
