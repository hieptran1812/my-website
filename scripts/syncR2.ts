/**
 * Sync local files to the site's R2 buckets through the gateway Worker
 * (infra/r2-worker), uploading only what changed.
 *
 *   tsx scripts/syncR2.ts media     public/imgs/**    → halleywiki-media  (key "imgs/…")
 *   tsx scripts/syncR2.ts content   content/blog/**   → halleywiki-content (key = path under content/blog)
 *
 * Images live only in R2 (served at NEXT_PUBLIC_MEDIA_BASE); the markdown stays
 * in git as the source of truth and is mirrored to R2 so serverless functions
 * can read a post without shipping the whole corpus (Vercel's 250 MB limit).
 *
 * Change detection compares each file's MD5 with the object's etag from one
 * listing per 1000 objects, so a build that adds three posts writes three
 * objects, not six thousand (R2 bills writes; the free tier is 1M/month).
 *
 * `content` also uploads the whole corpus as one brotli bundle
 * (CORPUS_BUNDLE_KEY) for the routes that walk every post.
 *
 * On Vercel, `content` runs for production builds only: the bucket is shared,
 * and a preview build must not overwrite what production serves.
 *
 * Needs R2_GATEWAY_URL and R2_GATEWAY_SECRET. Without them it skips and exits 0
 * (local builds, forks); with them, any failed upload exits 1 so a deploy never
 * ships pages whose content is missing from R2.
 */

import crypto from "crypto";
import fs from "fs";
import path from "path";
import zlib from "zlib";
import { CORPUS_BUNDLE_KEY, readLocalCorpus } from "../src/lib/contentSource";

const TARGETS = {
  media: { dir: "public/imgs", prefix: "imgs/" },
  content: { dir: "content/blog", prefix: "" },
} as const;
type Target = keyof typeof TARGETS;

const TYPES: Record<string, string> = {
  ".webp": "image/webp",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".svg": "image/svg+xml",
  ".avif": "image/avif",
  ".mp4": "video/mp4",
  ".md": "text/markdown; charset=utf-8",
  ".br": "application/octet-stream",
};

const CONCURRENCY = 16;

function walk(dir: string): string[] {
  const out: string[] = [];
  if (!fs.existsSync(dir)) return out;
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...walk(p));
    else if (e.isFile() && !e.name.startsWith(".")) out.push(p);
  }
  return out;
}

async function remoteEtags(base: string, auth: string, bucket: Target) {
  const etags = new Map<string, string>();
  let cursor: string | null = null;
  do {
    const url = `${base}/list/${bucket}${cursor ? `?cursor=${encodeURIComponent(cursor)}` : ""}`;
    const res = await fetch(url, { headers: { authorization: auth } });
    if (!res.ok) throw new Error(`list ${bucket}: HTTP ${res.status}`);
    const page = (await res.json()) as {
      objects: { key: string; etag: string }[];
      cursor: string | null;
    };
    for (const o of page.objects) etags.set(o.key, o.etag.replace(/"/g, ""));
    cursor = page.cursor;
  } while (cursor);
  return etags;
}

async function upload(
  base: string,
  auth: string,
  bucket: Target,
  key: string,
  file: string,
  data?: Buffer,
) {
  const body = data ?? fs.readFileSync(file);
  for (let attempt = 1; ; attempt++) {
    try {
      const res = await fetch(`${base}/${bucket}/${key.split("/").map(encodeURIComponent).join("/")}`, {
        method: "PUT",
        headers: {
          authorization: auth,
          "content-type": TYPES[path.extname(file).toLowerCase()] ?? "application/octet-stream",
        },
        body: new Uint8Array(body),
      });
      if (res.ok) return;
      throw new Error(`HTTP ${res.status}`);
    } catch (err) {
      if (attempt >= 4) throw new Error(`${key}: ${(err as Error).message}`);
      await new Promise((r) => setTimeout(r, 500 * 2 ** attempt));
    }
  }
}

async function main() {
  const bucket = process.argv[2] as Target;
  if (!(bucket in TARGETS)) {
    console.error("usage: tsx scripts/syncR2.ts media|content");
    process.exit(2);
  }
  const base = process.env.R2_GATEWAY_URL?.replace(/\/$/, "");
  const secret = process.env.R2_GATEWAY_SECRET;
  if (!base || !secret) {
    console.log(`☁️  R2 sync (${bucket}) skipped: R2_GATEWAY_URL/R2_GATEWAY_SECRET not set`);
    return;
  }
  if (bucket === "content" && process.env.VERCEL_ENV && process.env.VERCEL_ENV !== "production") {
    console.log(`☁️  R2 sync (content) skipped: ${process.env.VERCEL_ENV} build`);
    return;
  }
  const auth = `Bearer ${secret}`;
  const t0 = Date.now();
  const { dir, prefix } = TARGETS[bucket];

  const remote = await remoteEtags(base, auth, bucket);
  const todo: { key: string; file: string; body?: Buffer }[] = [];
  for (const file of walk(dir)) {
    const key = prefix + path.relative(dir, file).split(path.sep).join("/");
    const md5 = crypto.createHash("md5").update(fs.readFileSync(file)).digest("hex");
    if (remote.get(key) !== md5) todo.push({ key, file });
  }
  if (bucket === "content") {
    const body = zlib.brotliCompressSync(Buffer.from(JSON.stringify(readLocalCorpus())), {
      params: { [zlib.constants.BROTLI_PARAM_QUALITY]: 9 },
    });
    const md5 = crypto.createHash("md5").update(body).digest("hex");
    if (remote.get(CORPUS_BUNDLE_KEY) !== md5) {
      todo.push({ key: CORPUS_BUNDLE_KEY, file: "corpus.json.br", body });
    }
  }

  let done = 0;
  const failures: string[] = [];
  let next = 0;
  await Promise.all(
    Array.from({ length: CONCURRENCY }, async () => {
      while (next < todo.length) {
        const { key, file, body } = todo[next++];
        try {
          await upload(base, auth, bucket, key, file, body);
        } catch (err) {
          failures.push((err as Error).message);
        }
        if (++done % 500 === 0) console.log(`   …${done}/${todo.length}`);
      }
    }),
  );

  console.log(
    `☁️  R2 sync (${bucket}): ${todo.length - failures.length} uploaded, ` +
      `${remote.size} objects were already in R2, ${failures.length} failed ` +
      `in ${((Date.now() - t0) / 1000).toFixed(1)}s`,
  );
  if (failures.length) {
    console.error(failures.slice(0, 10).join("\n"));
    process.exit(1);
  }
}

main().catch((err) => {
  console.error("☁️  R2 sync failed:", err);
  process.exit(1);
});
