/**
 * Precompute the full-text blog search index (see src/lib/searchEngine.ts).
 *
 * Writes `src/lib/generated/searchIndex.json` (docs + vocabulary) and
 * `searchIndex.bin` (postings). The search route loads these once per server
 * instance instead of tokenizing ~170 MB of markdown on a cold start.
 *
 * Resilient by design: any failure removes the index and exits 0, and the
 * route falls back to building it in-process.
 *
 * Usage: tsx scripts/buildSearchIndex.ts
 */

import fs from "fs";
import path from "path";
import { loadAllPosts } from "../src/lib/blogIndex";
import { isPaywalledFile } from "../src/lib/paywall";
import {
  buildIndex,
  writeIndex,
  INDEX_BIN,
  INDEX_JSON,
} from "../src/lib/searchEngine";

async function main() {
  const t0 = Date.now();
  console.log("🔎 Building full-text search index…");
  try {
    const corpus = await loadAllPosts();
    const index = buildIndex(
      corpus,
      isPaywalledFile,
      path.join(process.cwd(), "content", "blog"),
    );
    const { jsonBytes, binBytes } = writeIndex(index);
    console.log(
      `✅ Indexed ${index.docs.length} posts, ${index.vocab.length} terms, ` +
        `${index.docIds.length} postings → searchIndex.json ` +
        `(${(jsonBytes / 1e6).toFixed(1)} MB) + searchIndex.bin ` +
        `(${(binBytes / 1e6).toFixed(1)} MB) in ${((Date.now() - t0) / 1000).toFixed(1)}s`,
    );
  } catch (err) {
    console.warn("⚠️  Search index build failed; route will index at runtime.", err);
    for (const f of [INDEX_JSON, INDEX_BIN]) fs.rmSync(f, { force: true });
  }
}

main();
