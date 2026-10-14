/**
 * Loads the search index once per server instance: the deploy-time files in
 * production, or an in-process build from the live corpus in dev (so new posts
 * are searchable without a rebuild) and whenever the files are missing.
 */

import path from "path";
import { loadAllPosts, type BlogIndexEntry } from "./blogIndex";
import { isPaywalledFile } from "./paywall";
import { buildIndex, readIndex, type SearchIndex } from "./searchEngine";

let generated: SearchIndex | null | undefined;
let built: { corpus: BlogIndexEntry[]; index: SearchIndex } | null = null;

export async function getSearchIndex(): Promise<SearchIndex> {
  if (process.env.NODE_ENV === "production") {
    if (generated === undefined) generated = readIndex();
    if (generated) return generated;
  }
  const corpus = await loadAllPosts();
  if (built?.corpus !== corpus) {
    built = {
      corpus,
      index: buildIndex(
        corpus,
        isPaywalledFile,
        path.join(process.cwd(), "content", "blog"),
      ),
    };
  }
  return built.index;
}
