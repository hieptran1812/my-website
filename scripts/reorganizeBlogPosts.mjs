#!/usr/bin/env node
/**
 * Reorganize blog posts so each markdown file lives at:
 *   content/blog/<category>/<subcategory>/<file>.md
 *
 * Category and subcategory are read from the post's frontmatter and slugified.
 * Already-correctly-placed files are skipped. Misplaced files are moved with
 * `git mv` so history is preserved.
 *
 * Usage:
 *   node scripts/reorganizeBlogPosts.mjs            # dry run (default)
 *   node scripts/reorganizeBlogPosts.mjs --apply    # actually move files
 */

import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import matter from "gray-matter";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, "..");
const BLOG_ROOT = path.join(REPO_ROOT, "content", "blog");
const APPLY = process.argv.includes("--apply");
const FORCE = process.argv.includes("--force");

function slugify(value) {
  if (typeof value !== "string") return "";
  return value
    .toLowerCase()
    .trim()
    .replace(/&/g, " and ")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function walk(dir) {
  const out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...walk(p));
    else if (entry.isFile() && entry.name.endsWith(".md")) out.push(p);
  }
  return out;
}

function ensureCleanWorktree() {
  const out = execFileSync("git", ["status", "--porcelain"], {
    cwd: REPO_ROOT,
    encoding: "utf8",
  });
  if (out.trim().length > 0) {
    console.error(
      "Refusing to run with a dirty working tree. Commit or stash first.",
    );
    console.error(out);
    process.exit(1);
  }
}

function isTracked(absPath) {
  try {
    execFileSync("git", ["ls-files", "--error-unmatch", absPath], {
      cwd: REPO_ROOT,
      stdio: "ignore",
    });
    return true;
  } catch {
    return false;
  }
}

function gitMove(src, dst) {
  fs.mkdirSync(path.dirname(dst), { recursive: true });
  if (isTracked(src)) {
    execFileSync("git", ["mv", src, dst], { cwd: REPO_ROOT, stdio: "inherit" });
  } else {
    // Untracked file: a plain rename keeps it untracked at the new location.
    fs.renameSync(src, dst);
  }
}

function main() {
  if (!fs.existsSync(BLOG_ROOT)) {
    console.error(`Blog root not found: ${BLOG_ROOT}`);
    process.exit(1);
  }

  if (APPLY && !FORCE) ensureCleanWorktree();

  const files = walk(BLOG_ROOT);
  const moves = [];
  const skipped = [];
  const missing = [];
  const relImageWarnings = [];

  for (const abs of files) {
    const raw = fs.readFileSync(abs, "utf8");
    let parsed;
    try {
      parsed = matter(raw);
    } catch (err) {
      missing.push({ file: abs, reason: `frontmatter parse error: ${err.message}` });
      continue;
    }
    const fm = parsed.data || {};
    const catSlug = slugify(fm.category);
    const subSlug = slugify(fm.subcategory);

    if (!catSlug || !subSlug) {
      missing.push({
        file: abs,
        reason: `missing ${!catSlug ? "category" : ""}${!catSlug && !subSlug ? "+" : ""}${!subSlug ? "subcategory" : ""}`,
      });
      continue;
    }

    const basename = path.basename(abs);
    const target = path.join(BLOG_ROOT, catSlug, subSlug, basename);

    if (path.resolve(target) === path.resolve(abs)) {
      skipped.push(abs);
      continue;
    }

    if (fs.existsSync(target)) {
      missing.push({
        file: abs,
        reason: `target already exists: ${path.relative(REPO_ROOT, target)}`,
      });
      continue;
    }

    if (/!\[[^\]]*\]\(\.{1,2}\//.test(parsed.content)) {
      relImageWarnings.push(path.relative(REPO_ROOT, abs));
    }

    moves.push({ from: abs, to: target });
  }

  console.log(`Total markdown files: ${files.length}`);
  console.log(`Already in place:     ${skipped.length}`);
  console.log(`Planned moves:        ${moves.length}`);
  console.log(`Skipped (review):     ${missing.length}`);
  if (relImageWarnings.length > 0) {
    console.log(
      `\n[WARN] ${relImageWarnings.length} file(s) use relative image paths (./ or ../) and may break after move:`,
    );
    for (const f of relImageWarnings.slice(0, 20)) console.log(`  - ${f}`);
    if (relImageWarnings.length > 20) console.log(`  ... and ${relImageWarnings.length - 20} more`);
  }
  if (missing.length > 0) {
    console.log("\n[REVIEW] Files needing manual attention:");
    for (const m of missing.slice(0, 50)) {
      console.log(`  - ${path.relative(REPO_ROOT, m.file)}  (${m.reason})`);
    }
    if (missing.length > 50) console.log(`  ... and ${missing.length - 50} more`);
  }

  if (moves.length === 0) {
    console.log("\nNothing to do.");
    return;
  }

  console.log("\nFirst 30 planned moves:");
  for (const m of moves.slice(0, 30)) {
    console.log(
      `  ${path.relative(REPO_ROOT, m.from)}  ->  ${path.relative(REPO_ROOT, m.to)}`,
    );
  }
  if (moves.length > 30) console.log(`  ... and ${moves.length - 30} more`);

  if (!APPLY) {
    console.log("\nDry run. Re-run with --apply to perform the moves.");
    return;
  }

  console.log("\nApplying moves...");
  for (const m of moves) {
    gitMove(m.from, m.to);
  }
  console.log(`Done. Moved ${moves.length} file(s).`);
}

main();
