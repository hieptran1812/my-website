/**
 * Cleaned post bodies for search snippets, precomputed at build time.
 *
 * The search route used to read each hit's markdown from content/blog, which
 * made Vercel trace the whole corpus (~180 MB) into the function and pushed it
 * past the 250 MB limit. Instead the build writes every public post's cleaned
 * body as its own brotli blob into `searchBodies.bin`, with a small offset
 * table in `searchBodies.json`; the route reads and inflates only the bodies
 * of the hits it returns.
 *
 * Paywalled posts are left out: their snippet is the fixed excerpt anyway.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import matter from "gray-matter";
import { cleanMarkdown, GENERATED_DIR, type SearchIndex } from "./searchEngine";

export const BODIES_JSON = path.join(GENERATED_DIR, "searchBodies.json");
export const BODIES_BIN = path.join(GENERATED_DIR, "searchBodies.bin");

/** file → [byte offset, byte length] into BODIES_BIN. */
type Table = Record<string, [number, number]>;

export function writeBodies(index: SearchIndex, blogDir: string) {
  const table: Table = {};
  const chunks: Buffer[] = [];
  let offset = 0;
  for (const doc of index.docs) {
    if (doc.paywalled) continue;
    let body = "";
    try {
      body = cleanMarkdown(
        matter(fs.readFileSync(path.join(blogDir, doc.file), "utf8")).content,
      );
    } catch {
      continue;
    }
    if (!body) continue;
    const blob = zlib.brotliCompressSync(Buffer.from(body, "utf8"), {
      params: { [zlib.constants.BROTLI_PARAM_QUALITY]: 9 },
    });
    table[doc.file] = [offset, blob.length];
    chunks.push(blob);
    offset += blob.length;
  }
  fs.writeFileSync(BODIES_BIN, Buffer.concat(chunks));
  fs.writeFileSync(BODIES_JSON, JSON.stringify(table));
  return { count: Object.keys(table).length, binBytes: offset };
}

let table: Table | null | undefined;
let fd: number | null = null;

function open(): boolean {
  if (table === undefined) {
    try {
      table = JSON.parse(fs.readFileSync(BODIES_JSON, "utf8")) as Table;
      fd = fs.openSync(BODIES_BIN, "r");
    } catch {
      table = null;
    }
  }
  return !!table && fd !== null;
}

/** The cleaned body of a post from the build-time store, or null if absent. */
export function readStoredBody(file: string): string | null {
  if (!open()) return null;
  const entry = table![file];
  if (!entry) return null;
  try {
    const buf = Buffer.alloc(entry[1]);
    fs.readSync(fd!, buf, 0, entry[1], entry[0]);
    return zlib.brotliDecompressSync(buf).toString("utf8");
  } catch {
    return null;
  }
}
