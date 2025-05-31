// Demo script to show how the read time calculation works
const fs = require("fs");
const matter = require("gray-matter");

// Simulate the calculateReadTimeWithTags function (simplified version)
function analyzeContent(content) {
  const cleanContent = content.replace(/^---[\s\S]*?---\n/, "");

  const textContent = cleanContent
    .replace(/```[\s\S]*?```/g, "")
    .replace(/`[^`]+`/g, "")
    .replace(/!\[.*?\]\(.*?\)/g, "")
    .replace(/\[.*?\]\(.*?\)/g, "")
    .replace(/#{1,6}\s+/g, "")
    .replace(/[*_]{1,2}(.*?)[*_]{1,2}/g, "$1")
    .replace(/[-*+]\s+/g, "")
    .replace(/\d+\.\s+/g, "")
    .replace(/\|.*?\|/g, "")
    .replace(/\$\$[\s\S]*?\$\$/g, "")
    .replace(/\$[^$]+\$/g, "")
    .replace(/\s+/g, " ")
    .trim();

  const wordCount = textContent
    .split(/\s+/)
    .filter((word) => word.length > 0).length;
  const codeBlocks = (cleanContent.match(/```[\s\S]*?```/g) || []).length;
  const mathEquations =
    (cleanContent.match(/\$\$[\s\S]*?\$\$/g) || []).length +
    (cleanContent.match(/\$[^$]+\$/g) || []).length;
  const images = (cleanContent.match(/!\[.*?\]\(.*?\)/g) || []).length;

  return {
    wordCount,
    codeBlocks,
    mathEquations,
    images,
    baseReadingTime: wordCount / 200, // 200 words per minute
  };
}

// Read the math-expressions-web.md file
const filePath = "./content/blog/math-expressions-web.md";
const fileContents = fs.readFileSync(filePath, "utf8");
const { data: frontmatter, content } = matter(fileContents);

const analysis = analyzeContent(content);

console.log("📖 Read Time Analysis for math-expressions-web.md:");
console.log("─────────────────────────────────────────────────");
console.log(`📄 Word count: ${analysis.wordCount} words`);
console.log(`💻 Code blocks: ${analysis.codeBlocks}`);
console.log(`🧮 Math equations: ${analysis.mathEquations}`);
console.log(`🖼️  Images: ${analysis.images}`);
console.log(
  `⏱️  Base reading time: ${analysis.baseReadingTime.toFixed(1)} minutes`
);
console.log(`⏱️  Final read time: ${frontmatter.readTime}`);
console.log("─────────────────────────────────────────────────");
