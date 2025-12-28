import fs from "fs/promises";
import path from "path";
import matter from "gray-matter";

// Node 18+ has global fetch; declare for TypeScript.
declare const fetch: (
  input: RequestInfo | URL,
  init?: RequestInit
) => Promise<Response>;

const PAPER_SUBFOLDERS = ["ai-interpretability", "ai-agent"];

type ArxivInfo = {
  url: string;
  id: string;
};

async function findArxivByTitle(title: string): Promise<ArxivInfo | null> {
  const trimmed = title.trim();
  if (!trimmed) return null;

  const query = encodeURIComponent(`ti:"${trimmed}"`);
  const apiUrl = `https://export.arxiv.org/api/query?search_query=${query}&max_results=3`;

  try {
    const res = await fetch(apiUrl, {
      headers: {
        "User-Agent": "my-website-ref-updater/1.0 (+https://github.com/)",
      },
    });
    if (!res.ok) {
      console.warn(`arXiv request failed for title '${title}': ${res.status}`);
      return null;
    }

    const text = await res.text();

    // Find the first <entry> that looks like an arXiv article.
    const entryMatch = text.match(
      /<entry>[\s\S]*?<id>https?:\/\/arxiv.org\/abs\/([^<]+)<\/id>[\s\S]*?<title>([\s\S]*?)<\/title>/i
    );
    if (!entryMatch) {
      console.warn(`No arXiv entry found for title '${title}'`);
      return null;
    }

    const arxivId = entryMatch[1].trim();
    const url = `https://arxiv.org/abs/${arxivId}`;
    return { url, id: arxivId };
  } catch (err) {
    console.warn(`Error querying arXiv for title '${title}':`, err);
    return null;
  }
}

function normalizeOpenReviewUrl(rawUrl: string): string {
  let url = rawUrl.trim();

  // Convert pdf?id=NOTE_ID to forum?id=NOTE_ID
  url = url.replace("openreview.net/pdf?id=", "openreview.net/forum?id=");

  // Convert pdf/<hash>.pdf to forum?id=<hash>
  const pdfPathMatch = url.match(
    /https:\/\/openreview\.net\/pdf\/([a-f0-9]+)\.pdf/i
  );
  if (pdfPathMatch) {
    const id = pdfPathMatch[1];
    url = `https://openreview.net/forum?id=${id}`;
  }

  return url;
}

async function processFile(filePath: string): Promise<void> {
  const raw = await fs.readFile(filePath, "utf8");
  const parsed = matter(raw);
  const title = String(parsed.data.title ?? "").trim();

  if (!title) {
    console.warn(
      `Skipping file without title: ${path.relative(process.cwd(), filePath)}`
    );
    return;
  }

  let content = parsed.content.trimEnd() + "\n";
  const refsMarker = "## References";

  let beforeRefs = content;
  let refsBody = "\n\n";

  if (content.includes(refsMarker)) {
    const parts = content.split(refsMarker);
    beforeRefs = parts[0];
    refsBody = parts[1] ?? "\n\n";
  } else {
    // Ensure a blank line before new References section
    if (!beforeRefs.endsWith("\n\n")) {
      beforeRefs = beforeRefs.trimEnd() + "\n\n";
    }
  }

  const existingOpenReviewMatch = refsBody.match(
    /https:\/\/openreview\.net[^)\s>]*/
  );
  const existingOpenReviewUrl = existingOpenReviewMatch
    ? existingOpenReviewMatch[0]
    : null;

  const normalizedOpenReviewUrl = existingOpenReviewUrl
    ? normalizeOpenReviewUrl(existingOpenReviewUrl)
    : null;

  // Collect any additional reference lines that are not openreview/arxiv links.
  const originalLines = refsBody
    .split("\n")
    .map((l) => l.trimEnd())
    .filter((l) => l.trim() !== "");

  const additionalLines = originalLines.filter(
    (line) =>
      !line.includes("openreview.net") && !line.includes("arxiv.org/abs")
  );

  const newRefLines: string[] = [];
  let counter = 1;

  if (normalizedOpenReviewUrl) {
    newRefLines.push(`${counter}. [${title}](${normalizedOpenReviewUrl})`);
    counter += 1;
  }

  const arxivInfo = await findArxivByTitle(title);

  if (arxivInfo) {
    newRefLines.push(
      `${counter}. [${title} (arXiv:${arxivInfo.id})](${arxivInfo.url})`
    );
    counter += 1;
  } else if (!normalizedOpenReviewUrl) {
    // No openreview and no arxiv found: keep whatever was there before.
    if (originalLines.length === 0) {
      console.warn(
        `No references and no arXiv found for '${title}' in ${path.basename(
          filePath
        )}`
      );
    } else {
      // Reuse original reference lines (renumbered).
      for (const line of originalLines) {
        const cleaned = line.replace(/^\s*\d+\.\s*/, "");
        newRefLines.push(`${counter}. ${cleaned}`);
        counter += 1;
      }
    }
  }

  // Append any additional, non-arxiv/openreview references from original section.
  for (const line of additionalLines) {
    const cleaned = line.replace(/^\s*\d+\.\s*/, "");
    newRefLines.push(`${counter}. ${cleaned}`);
    counter += 1;
  }

  if (newRefLines.length === 0) {
    // Nothing to change.
    return;
  }

  const newContent = `${beforeRefs}${refsMarker}\n\n${newRefLines.join(
    "\n"
  )}\n`;

  const updated = matter.stringify(newContent, parsed.data);
  await fs.writeFile(filePath, updated, "utf8");
  console.log(
    `Updated references for: ${path.relative(process.cwd(), filePath)}`
  );
}

async function main() {
  const rootBase = path.join(process.cwd(), "content", "blog", "paper-reading");

  for (const subfolder of PAPER_SUBFOLDERS) {
    const dirPath = path.join(rootBase, subfolder);

    let entries;
    try {
      entries = await fs.readdir(dirPath, { withFileTypes: true });
    } catch (err) {
      console.warn(`Skip missing folder: ${dirPath}`);
      continue;
    }

    const mdFiles = entries
      .filter((e) => e.isFile() && e.name.endsWith(".md"))
      .map((e) => path.join(dirPath, e.name));

    for (const file of mdFiles) {
      try {
        await processFile(file);
      } catch (err) {
        console.error(
          `Error processing ${path.relative(process.cwd(), file)}:`,
          err
        );
      }
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
