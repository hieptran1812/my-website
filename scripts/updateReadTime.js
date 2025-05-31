#!/usr/bin/env node

/**
 * Script to automatically calculate and update readTime for all blog posts
 * Usage: npm run update-readtime or node scripts/updateReadTime.js
 */

import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { fileURLToPath } from "url";
import { calculateReadTimeWithTags } from "../src/lib/readTimeCalculator.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const BLOG_CONTENT_DIR = path.join(__dirname, "..", "content", "blog");

async function updateReadTimes() {
  console.log("🚀 Starting automatic readTime calculation for blog posts...\n");

  if (!fs.existsSync(BLOG_CONTENT_DIR)) {
    console.error("❌ Blog content directory not found:", BLOG_CONTENT_DIR);
    process.exit(1);
  }

  const files = fs
    .readdirSync(BLOG_CONTENT_DIR)
    .filter((file) => file.endsWith(".md"));

  if (files.length === 0) {
    console.log("📁 No markdown files found in blog directory");
    return;
  }

  console.log(`📄 Found ${files.length} blog posts to process\n`);

  let updatedCount = 0;
  let skippedCount = 0;

  for (const filename of files) {
    const filePath = path.join(BLOG_CONTENT_DIR, filename);

    try {
      const fileContents = fs.readFileSync(filePath, "utf8");
      const { data: frontmatter, content } = matter(fileContents);

      // Calculate new readTime
      const readTimeResult = calculateReadTimeWithTags(
        content,
        frontmatter.tags || [],
        frontmatter.category || "general"
      );

      const oldReadTime = frontmatter.readTime;
      const newReadTime = readTimeResult.readTime;

      // Check if update is needed
      if (oldReadTime !== newReadTime) {
        // Update frontmatter
        frontmatter.readTime = newReadTime;

        // Reconstruct the file with updated frontmatter
        const updatedFile = matter.stringify(content, frontmatter);

        // Write back to file
        fs.writeFileSync(filePath, updatedFile, "utf8");

        console.log(`✅ ${filename}`);
        console.log(
          `   Old: ${oldReadTime || "not set"} → New: ${newReadTime}`
        );
        console.log(
          `   Analysis: ${readTimeResult.analysis.wordCount} words, ${readTimeResult.analysis.complexity} complexity`
        );
        console.log("");

        updatedCount++;
      } else {
        console.log(
          `⏭️  ${filename} - readTime already correct (${oldReadTime})`
        );
        skippedCount++;
      }
    } catch (error) {
      console.error(`❌ Error processing ${filename}:`, error.message);
    }
  }

  console.log("\n📊 Summary:");
  console.log(`   ✅ Updated: ${updatedCount} files`);
  console.log(`   ⏭️  Skipped: ${skippedCount} files`);
  console.log(`   📄 Total: ${files.length} files`);

  if (updatedCount > 0) {
    console.log(
      "\n🎉 ReadTime calculation complete! All blog posts have been updated."
    );
  } else {
    console.log("\n✨ All blog posts already have correct readTime values.");
  }
}

// For CLI usage
if (import.meta.url === `file://${process.argv[1]}`) {
  updateReadTimes().catch(console.error);
}

export { updateReadTimes };
