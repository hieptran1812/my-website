/**
 * Move the precomputed related-articles graph between CI and the Vercel build.
 *
 * Computing the graph (scripts/buildRelatedGraph.ts) took 15–19 min of every
 * Vercel build on its 2-core builder. It now runs in GitHub Actions
 * (.github/workflows/related-graph.yml) on every push to main, which uploads
 * the result here; the Vercel build only downloads it.
 *
 *   tsx scripts/relatedGraphR2.ts put     upload src/lib/generated/blogGraph.br
 *                                        as _graph/latest.br + _graph/latest.sha
 *   tsx scripts/relatedGraphR2.ts fetch   download it into src/lib/generated/
 *
 * `fetch` waits (up to GRAPH_WAIT_SECONDS, default 600) until the uploaded
 * graph was computed from this very commit (VERCEL_GIT_COMMIT_SHA), so new
 * posts get their graph on the deploy that adds them. If CI is late it takes
 * the latest graph there is; if there is none, or R2 is unreachable, it runs
 * buildRelatedGraph locally as before. It never fails the build.
 *
 * Objects live in the private content bucket via the gateway Worker
 * (infra/r2-worker); needs R2_GATEWAY_URL and R2_GATEWAY_SECRET.
 */

import { execSync } from "child_process";
import fs from "fs";
import path from "path";

const OUT = path.join(process.cwd(), "src", "lib", "generated", "blogGraph.br");
const KEY_GRAPH = "_graph/latest.br";
const KEY_SHA = "_graph/latest.sha";

const base = (process.env.R2_GATEWAY_URL || "").replace(/\/$/, "");
const auth = `Bearer ${process.env.R2_GATEWAY_SECRET || ""}`;

async function get(key: string): Promise<Buffer | null> {
  const res = await fetch(`${base}/content/${key}`, {
    headers: { authorization: auth },
  });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`GET ${key}: HTTP ${res.status}`);
  return Buffer.from(await res.arrayBuffer());
}

async function put(key: string, body: Buffer, type: string): Promise<void> {
  const res = await fetch(`${base}/content/${key}`, {
    method: "PUT",
    headers: { authorization: auth, "content-type": type },
    body: new Uint8Array(body),
  });
  if (!res.ok) throw new Error(`PUT ${key}: HTTP ${res.status}`);
}

function buildLocally(reason: string): void {
  console.log(`🕸️  ${reason} — computing the related graph in this build instead`);
  execSync("npx tsx scripts/buildRelatedGraph.ts", { stdio: "inherit" });
}

async function doPut(): Promise<void> {
  const sha = process.env.GITHUB_SHA || process.env.GRAPH_SHA;
  if (!sha) throw new Error("GITHUB_SHA (or GRAPH_SHA) is required");
  const graph = fs.readFileSync(OUT);
  // Graph first, then the sha that marks it ready: a reader that sees the new
  // sha always gets the matching graph.
  await put(KEY_GRAPH, graph, "application/octet-stream");
  await put(KEY_SHA, Buffer.from(sha), "text/plain");
  console.log(`☁️  Uploaded related graph (${(graph.length / 1e6).toFixed(1)} MB) for ${sha.slice(0, 7)}`);
}

async function doFetch(): Promise<void> {
  if (!base || !process.env.R2_GATEWAY_SECRET) {
    buildLocally("R2 not configured");
    return;
  }
  const want = process.env.VERCEL_GIT_COMMIT_SHA || "";
  const waitMs = Number(process.env.GRAPH_WAIT_SECONDS ?? 600) * 1000;
  const t0 = Date.now();
  try {
    let sha = "";
    for (;;) {
      sha = (await get(KEY_SHA))?.toString("utf8").trim() ?? "";
      if (!want || sha === want || Date.now() - t0 >= waitMs) break;
      await new Promise((r) => setTimeout(r, 15_000));
    }
    const graph = await get(KEY_GRAPH);
    if (!graph) {
      buildLocally("no related graph in R2 yet");
      return;
    }
    fs.mkdirSync(path.dirname(OUT), { recursive: true });
    fs.writeFileSync(OUT, graph);
    const waited = ((Date.now() - t0) / 1000).toFixed(0);
    console.log(
      sha === want || !want
        ? `🕸️  Related graph for ${sha.slice(0, 7)} from R2 (waited ${waited}s)`
        : `🕸️  ⚠️  CI graph for this commit not ready after ${waited}s; using the latest one (${sha.slice(0, 7)})`,
    );
  } catch (err) {
    buildLocally(`R2 graph fetch failed (${(err as Error).message})`);
  }
}

const mode = process.argv[2];
(mode === "put" ? doPut() : mode === "fetch" ? doFetch() : Promise.reject(new Error("usage: put|fetch"))).catch(
  (err) => {
    console.error("relatedGraphR2:", err);
    process.exit(mode === "put" ? 1 : 0);
  },
);
