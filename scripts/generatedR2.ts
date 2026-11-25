/**
 * Move build-time generated files between CI and the Vercel build.
 *
 * Precomputing them took ~4 min of every Vercel build on its 2-core builder
 * (related graph, blog index, search index, R2 content sync), one after the
 * other. They now run in GitHub Actions (.github/workflows/related-graph.yml)
 * on every push to main, as two parallel jobs that upload their outputs here;
 * the Vercel build only downloads them.
 *
 *   tsx scripts/generatedR2.ts put <set>   upload <set>'s files, then its sha
 *   tsx scripts/generatedR2.ts fetch       download every set into src/lib/generated/
 *
 * Sets:
 *   graph    blogGraph.br                                   (buildRelatedGraph)
 *   indexes  blogPostsIndex.json, searchIndex.*, searchBodies.*
 *            (buildBlogIndex + buildSearchIndex; the job also runs the R2
 *            content sync before uploading, so a matching sha means the
 *            markdown for this commit is in R2 too)
 *
 * `fetch` waits (up to GRAPH_WAIT_SECONDS, default 600) until each set was
 * computed from this very commit (VERCEL_GIT_COMMIT_SHA). If CI is late, or
 * R2 is unreachable, it builds that set locally as the build used to; the
 * graph alone may fall back to the latest one in R2, since a slightly stale
 * related-posts graph is harmless while a stale index would hide new posts.
 * Preview builds (no matching CI run) build the indexes locally and take the
 * latest graph. It fails the build only if a local fallback fails.
 *
 * Objects live in the private content bucket under _gen/ via the gateway
 * Worker (infra/r2-worker); needs R2_GATEWAY_URL and R2_GATEWAY_SECRET.
 */

import { execSync } from "child_process";
import fs from "fs";
import path from "path";

const GEN = path.join(process.cwd(), "src", "lib", "generated");

const SETS = {
  graph: {
    files: ["blogGraph.br"],
    build: ["npx tsx scripts/buildRelatedGraph.ts"],
    staleOk: true,
  },
  indexes: {
    files: [
      "blogPostsIndex.json",
      "searchIndex.json",
      "searchIndex.bin",
      "searchBodies.json",
      "searchBodies.bin",
    ],
    build: [
      "npx tsx scripts/buildBlogIndex.ts",
      "npx tsx scripts/buildSearchIndex.ts",
      "npx tsx scripts/syncR2.ts content",
    ],
    staleOk: false,
  },
} as const;
type SetName = keyof typeof SETS;

const base = (process.env.R2_GATEWAY_URL || "").replace(/\/$/, "");
const auth = `Bearer ${process.env.R2_GATEWAY_SECRET || ""}`;
const key = (set: string, file: string) => `_gen/${set}/${file}`;

async function get(k: string): Promise<Buffer | null> {
  const res = await fetch(`${base}/content/${k}`, { headers: { authorization: auth } });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`GET ${k}: HTTP ${res.status}`);
  return Buffer.from(await res.arrayBuffer());
}

async function put(k: string, body: Buffer): Promise<void> {
  for (let attempt = 1; ; attempt++) {
    const res = await fetch(`${base}/content/${k}`, {
      method: "PUT",
      headers: { authorization: auth, "content-type": "application/octet-stream" },
      body: new Uint8Array(body),
    }).catch((err: Error) => err);
    if (!(res instanceof Error) && res.ok) return;
    if (attempt >= 4) throw new Error(`PUT ${k}: ${res instanceof Error ? res.message : `HTTP ${res.status}`}`);
    await new Promise((r) => setTimeout(r, 1000 * attempt));
  }
}

async function doPut(set: SetName): Promise<void> {
  const sha = process.env.GITHUB_SHA || process.env.GRAPH_SHA;
  if (!sha) throw new Error("GITHUB_SHA (or GRAPH_SHA) is required");
  let bytes = 0;
  // Files first, then the sha that marks them ready: a reader that sees the
  // new sha always gets the matching files.
  await Promise.all(
    SETS[set].files.map(async (f) => {
      const body = fs.readFileSync(path.join(GEN, f));
      bytes += body.length;
      await put(key(set, f), body);
    }),
  );
  await put(key(set, "sha"), Buffer.from(sha));
  console.log(`☁️  Uploaded ${set} (${(bytes / 1e6).toFixed(1)} MB) for ${sha.slice(0, 7)}`);
}

function buildLocally(set: SetName, reason: string): void {
  console.log(`🛠️  ${set}: ${reason} — building it in this build instead`);
  for (const cmd of SETS[set].build) execSync(cmd, { stdio: "inherit" });
}

async function download(set: SetName): Promise<boolean> {
  const bodies = await Promise.all(SETS[set].files.map((f) => get(key(set, f))));
  if (bodies.some((b) => !b)) return false;
  fs.mkdirSync(GEN, { recursive: true });
  SETS[set].files.forEach((f, i) => fs.writeFileSync(path.join(GEN, f), bodies[i]!));
  return true;
}

/** Downloads a set; returns why it must be built locally instead, or null. */
async function fetchSet(set: SetName, want: string, waitMs: number, t0: number): Promise<string | null> {
  const { staleOk } = SETS[set];
  if (!want) {
    // Preview / non-CI build: nothing was computed for this commit.
    if (staleOk && (await download(set))) {
      console.log(`📦 ${set}: latest from R2 (preview build)`);
      return null;
    }
    return "preview build";
  }
  let sha = "";
  for (;;) {
    sha = (await get(key(set, "sha")))?.toString("utf8").trim() ?? "";
    if (sha === want || Date.now() - t0 >= waitMs) break;
    await new Promise((r) => setTimeout(r, 10_000));
  }
  const waited = ((Date.now() - t0) / 1000).toFixed(0);
  if (sha !== want && !staleOk) return `CI result for ${want.slice(0, 7)} not ready after ${waited}s`;
  if (!(await download(set))) return "not in R2";
  console.log(
    sha === want
      ? `📦 ${set} for ${sha.slice(0, 7)} from R2 (waited ${waited}s)`
      : `📦 ⚠️  ${set}: CI result for this commit not ready after ${waited}s; using the latest (${sha.slice(0, 7)})`,
  );
  return null;
}

async function doFetch(): Promise<void> {
  const sets = Object.keys(SETS) as SetName[];
  if (!base || !process.env.R2_GATEWAY_SECRET) {
    for (const s of sets) buildLocally(s, "R2 not configured");
    return;
  }
  const production = !process.env.VERCEL_ENV || process.env.VERCEL_ENV === "production";
  const want = production ? process.env.VERCEL_GIT_COMMIT_SHA || "" : "";
  const waitMs = Number(process.env.GRAPH_WAIT_SECONDS ?? 600) * 1000;
  const t0 = Date.now();
  // Indexes first: the graph is the slower CI job, so it is usually ready by
  // the time the indexes are. Sequential keeps local fallbacks from competing
  // for the 2-core builder.
  for (const s of ["indexes", "graph"] as SetName[]) {
    let reason: string | null;
    try {
      reason = await fetchSet(s, want, waitMs, t0);
    } catch (err) {
      reason = `R2 fetch failed (${(err as Error).message})`;
    }
    // Outside the try: a failing local build (e.g. the content sync) must
    // fail the deploy, as it did before.
    if (reason) buildLocally(s, reason);
  }
}

const [mode, set] = process.argv.slice(2);
const run =
  mode === "put" && set in SETS
    ? doPut(set as SetName)
    : mode === "fetch"
      ? doFetch()
      : Promise.reject(new Error("usage: put graph|indexes | fetch"));
run.catch((err) => {
  console.error("generatedR2:", err);
  process.exit(1);
});
