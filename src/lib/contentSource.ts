/**
 * Where post markdown is read from at runtime.
 *
 * The markdown lives in git (content/blog) and the build reads it from disk.
 * Serverless functions do NOT ship it: next.config excludes content/** from
 * every function, because the ~180 MB corpus pushed them past Vercel's 250 MB
 * limit. The build mirrors the corpus into the private R2 bucket
 * halleywiki-content (scripts/syncR2.ts) and functions read it from there
 * through the gateway Worker (infra/r2-worker):
 *
 *   - one post      → GET /content/<slug>.md
 *   - whole corpus  → GET /content/_bundle/corpus.json.br (one object, brotli),
 *                     for the few routes that walk every post
 *
 * Whenever content/blog exists on disk (dev, the build itself) it is used
 * directly, so local behaviour is unchanged.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";

export const BLOG_DIR = path.join(process.cwd(), "content", "blog");
export const CORPUS_BUNDLE_KEY = "_bundle/corpus.json.br";

const GATEWAY = (process.env.R2_GATEWAY_URL || "").replace(/\/$/, "");
const SECRET = process.env.R2_GATEWAY_SECRET || "";
const TTL_MS = 5 * 60 * 1000;

let localCorpus: boolean | undefined;

/** True when the markdown corpus is on disk (dev and build). */
export function hasLocalCorpus(): boolean {
  // CONTENT_SOURCE=r2 forces the R2 path even with content/ on disk (testing).
  if (process.env.CONTENT_SOURCE === "r2") return false;
  if (localCorpus === undefined) {
    try {
      localCorpus = fs.statSync(BLOG_DIR).isDirectory();
    } catch {
      localCorpus = false;
    }
  }
  return localCorpus;
}

/** "a/b/c" → "a/b/c.md", or null for anything that could escape the corpus. */
export function slugToRelPath(slug: string): string | null {
  const parts = slug.split("/").filter(Boolean);
  if (parts.length === 0) return null;
  if (parts.some((p) => p === "." || p === ".." || p.includes("\\"))) return null;
  return parts.join("/") + ".md";
}

async function gatewayGet(key: string): Promise<Response | null> {
  if (!GATEWAY || !SECRET) return null;
  const res = await fetch(
    `${GATEWAY}/content/${key.split("/").map(encodeURIComponent).join("/")}`,
    // Not `cache: "no-store"`: that marks an ISR render as dynamic and Next
    // throws DYNAMIC_SERVER_USAGE (500) for every on-demand post.
    { headers: { authorization: `Bearer ${SECRET}` }, next: { revalidate: 3600 } },
  );
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`R2 content ${key}: HTTP ${res.status}`);
  return res;
}

const postCache = new Map<string, { raw: string | null; at: number }>();

/**
 * Raw markdown of one post by its path under content/blog ("a/b.md"), or null
 * if there is no such post.
 */
export async function readPostRaw(relPath: string): Promise<string | null> {
  if (hasLocalCorpus()) {
    try {
      return fs.readFileSync(path.join(BLOG_DIR, relPath), "utf8");
    } catch {
      return null;
    }
  }
  const hit = postCache.get(relPath);
  if (hit && Date.now() - hit.at < TTL_MS) return hit.raw;
  const res = await gatewayGet(relPath);
  const raw = res ? await res.text() : null;
  postCache.set(relPath, { raw, at: Date.now() });
  return raw;
}

let bundle: { files: Record<string, string>; at: number } | null = null;
let bundleInFlight: Promise<Record<string, string>> | null = null;

/**
 * Every post as { relPath: raw markdown }. Reads the disk when the corpus is
 * there; otherwise downloads the corpus bundle once per TTL.
 */
export async function readCorpus(): Promise<Record<string, string>> {
  if (hasLocalCorpus()) return readLocalCorpus();
  if (bundle && Date.now() - bundle.at < TTL_MS) return bundle.files;
  if (bundleInFlight) return bundleInFlight;
  bundleInFlight = (async () => {
    try {
      const res = await gatewayGet(CORPUS_BUNDLE_KEY);
      if (!res) throw new Error("R2 corpus bundle missing");
      const buf = Buffer.from(await res.arrayBuffer());
      const files = JSON.parse(zlib.brotliDecompressSync(buf).toString("utf8"));
      bundle = { files, at: Date.now() };
      return files as Record<string, string>;
    } finally {
      bundleInFlight = null;
    }
  })();
  return bundleInFlight;
}

/** The corpus on disk as { relPath: raw markdown } (used by the build too). */
export function readLocalCorpus(): Record<string, string> {
  const out: Record<string, string> = {};
  const stack = [BLOG_DIR];
  while (stack.length > 0) {
    const dir = stack.pop()!;
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const e of entries) {
      const full = path.join(dir, e.name);
      if (e.isDirectory()) stack.push(full);
      else if (e.isFile() && e.name.endsWith(".md")) {
        try {
          out[path.relative(BLOG_DIR, full).split(path.sep).join("/")] =
            fs.readFileSync(full, "utf8");
        } catch {
          /* skip unreadable file */
        }
      }
    }
  }
  return out;
}
