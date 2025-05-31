/**
 * Advanced Read Time Integration Script
 * This script demonstrates how to integrate the read time calculator
 * into your blog workflow and development process
 */

import fs from "fs";
import path from "path";
import matter from "gray-matter";
import {
  calculateReadTimeWithTags,
  ReadTimeResult,
} from "../src/lib/readTimeCalculator";

interface BlogPost {
  filename: string;
  title: string;
  readTime: string;
  wordCount: number;
  complexity: string;
  tags: string[];
  category: string;
  publishDate: string;
}

/**
 * Get analytics for all blog posts
 */
export async function getBlogAnalytics(): Promise<{
  posts: BlogPost[];
  totalPosts: number;
  averageReadTime: number;
  totalWords: number;
  complexityDistribution: Record<string, number>;
}> {
  const BLOG_DIR = path.join(process.cwd(), "content", "blog");
  const files = fs.readdirSync(BLOG_DIR).filter((file) => file.endsWith(".md"));

  const posts: BlogPost[] = [];
  let totalWords = 0;
  let totalReadTimeMinutes = 0;
  const complexityCount: Record<string, number> = {
    beginner: 0,
    intermediate: 0,
    advanced: 0,
  };

  for (const filename of files) {
    const filePath = path.join(BLOG_DIR, filename);
    const fileContents = fs.readFileSync(filePath, "utf8");
    const { data: frontmatter, content } = matter(fileContents);

    const readTimeResult = calculateReadTimeWithTags(
      content,
      frontmatter.tags || [],
      frontmatter.category || "general"
    );

    const post: BlogPost = {
      filename,
      title: frontmatter.title || "Untitled",
      readTime: readTimeResult.readTime,
      wordCount: readTimeResult.analysis.wordCount,
      complexity: readTimeResult.analysis.complexity,
      tags: frontmatter.tags || [],
      category: frontmatter.category || "general",
      publishDate: frontmatter.publishDate || "",
    };

    posts.push(post);
    totalWords += readTimeResult.analysis.wordCount;
    totalReadTimeMinutes += readTimeResult.readTimeMinutes;
    complexityCount[readTimeResult.analysis.complexity]++;
  }

  return {
    posts,
    totalPosts: posts.length,
    averageReadTime: Math.round(totalReadTimeMinutes / posts.length),
    totalWords,
    complexityDistribution: complexityCount,
  };
}

/**
 * Generate a blog content report
 */
export async function generateBlogReport(): Promise<void> {
  console.log("📊 Generating Blog Content Report...\n");

  const analytics = await getBlogAnalytics();

  console.log("=".repeat(60));
  console.log("📈 BLOG CONTENT ANALYTICS REPORT");
  console.log("=".repeat(60));

  console.log(`\n📚 Overview:`);
  console.log(`   Total Posts: ${analytics.totalPosts}`);
  console.log(`   Total Words: ${analytics.totalWords.toLocaleString()}`);
  console.log(`   Average Read Time: ${analytics.averageReadTime} minutes`);
  console.log(
    `   Average Words per Post: ${Math.round(
      analytics.totalWords / analytics.totalPosts
    )}`
  );

  console.log(`\n🎯 Complexity Distribution:`);
  Object.entries(analytics.complexityDistribution).forEach(
    ([complexity, count]) => {
      const percentage = ((count / analytics.totalPosts) * 100).toFixed(1);
      console.log(
        `   ${
          complexity.charAt(0).toUpperCase() + complexity.slice(1)
        }: ${count} posts (${percentage}%)`
      );
    }
  );

  console.log(`\n📋 Post Details:`);
  analytics.posts
    .sort(
      (a, b) =>
        new Date(b.publishDate).getTime() - new Date(a.publishDate).getTime()
    )
    .forEach((post) => {
      console.log(`   📄 ${post.title}`);
      console.log(
        `      Read Time: ${post.readTime} | Words: ${post.wordCount} | Complexity: ${post.complexity}`
      );
      console.log(
        `      Category: ${post.category} | Tags: ${post.tags.join(", ")}`
      );
      console.log("");
    });

  console.log("=".repeat(60));
  console.log("Report generated successfully! ✅");
}

/**
 * Validate read times across all posts
 */
export async function validateReadTimes(): Promise<void> {
  console.log("🔍 Validating read times across all blog posts...\n");

  const analytics = await getBlogAnalytics();
  let issuesFound = 0;

  for (const post of analytics.posts) {
    const filePath = path.join(process.cwd(), "content", "blog", post.filename);
    const fileContents = fs.readFileSync(filePath, "utf8");
    const { data: frontmatter, content } = matter(fileContents);

    const calculatedReadTime = calculateReadTimeWithTags(
      content,
      frontmatter.tags || [],
      frontmatter.category || "general"
    );

    if (frontmatter.readTime !== calculatedReadTime.readTime) {
      console.log(`⚠️  Mismatch in ${post.filename}:`);
      console.log(`   Current: ${frontmatter.readTime}`);
      console.log(`   Should be: ${calculatedReadTime.readTime}`);
      console.log("");
      issuesFound++;
    }
  }

  if (issuesFound === 0) {
    console.log("✅ All read times are correctly calculated!");
  } else {
    console.log(`❌ Found ${issuesFound} posts with incorrect read times.`);
    console.log('💡 Run "npm run update-readtime" to fix these issues.');
  }
}

// CLI interface
if (require.main === module) {
  const command = process.argv[2];

  switch (command) {
    case "report":
      generateBlogReport();
      break;
    case "validate":
      validateReadTimes();
      break;
    default:
      console.log("📖 Blog Read Time Tools");
      console.log("Usage:");
      console.log(
        "  npx tsx scripts/blogAnalytics.ts report    - Generate content analytics report"
      );
      console.log(
        "  npx tsx scripts/blogAnalytics.ts validate  - Validate read times"
      );
  }
}
