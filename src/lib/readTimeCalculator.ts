/**
 * Automatic Read Time Calculator
 * Calculates reading time based on content analysis and complexity factors
 */

export interface ReadTimeConfig {
  /** Average words per minute for reading (default: 200) */
  wordsPerMinute?: number;
  /** Additional time for code blocks in seconds per block (default: 30) */
  codeBlockTime?: number;
  /** Additional time for math equations in seconds per equation (default: 15) */
  mathEquationTime?: number;
  /** Additional time for images in seconds per image (default: 12) */
  imageTime?: number;
  /** Additional time for tables in seconds per table (default: 20) */
  tableTime?: number;
  /** Minimum read time in minutes (default: 1) */
  minReadTime?: number;
  /** Maximum read time in minutes for reasonable limits (default: 60) */
  maxReadTime?: number;
}

export interface ContentAnalysis {
  wordCount: number;
  codeBlocks: number;
  mathEquations: number;
  images: number;
  tables: number;
  headings: number;
  lists: number;
  links: number;
  complexity: "beginner" | "intermediate" | "advanced";
}

export interface ReadTimeResult {
  readTime: string;
  readTimeMinutes: number;
  analysis: ContentAnalysis;
  breakdown: {
    baseReadingTime: number;
    codeBlockTime: number;
    mathTime: number;
    imageTime: number;
    tableTime: number;
    complexityAdjustment: number;
    totalTime: number;
  };
}

const DEFAULT_CONFIG: Required<ReadTimeConfig> = {
  wordsPerMinute: 200,
  codeBlockTime: 30,
  mathEquationTime: 15,
  imageTime: 12,
  tableTime: 20,
  minReadTime: 1,
  maxReadTime: 60,
};

/**
 * Analyzes markdown content to extract various content types
 */
export function analyzeContent(content: string): ContentAnalysis {
  // Remove frontmatter if present
  const cleanContent = content.replace(/^---[\s\S]*?---\n/, "");

  // Count words (excluding code blocks and other special content)
  const textContent = cleanContent
    .replace(/```[\s\S]*?```/g, "") // Remove code blocks
    .replace(/`[^`]+`/g, "") // Remove inline code
    .replace(/!\[.*?\]\(.*?\)/g, "") // Remove images
    .replace(/\[.*?\]\(.*?\)/g, "") // Remove links (keep text only)
    .replace(/#{1,6}\s+/g, "") // Remove heading markers
    .replace(/[*_]{1,2}(.*?)[*_]{1,2}/g, "$1") // Remove emphasis markers
    .replace(/[-*+]\s+/g, "") // Remove list markers
    .replace(/\d+\.\s+/g, "") // Remove numbered list markers
    .replace(/\|.*?\|/g, "") // Remove table syntax
    .replace(/\$\$[\s\S]*?\$\$/g, "") // Remove block math
    .replace(/\$[^$]+\$/g, "") // Remove inline math
    .replace(/\s+/g, " ") // Normalize whitespace
    .trim();

  const wordCount = textContent
    .split(/\s+/)
    .filter((word) => word.length > 0).length;

  // Count different content types
  const codeBlocks = (cleanContent.match(/```[\s\S]*?```/g) || []).length;
  const inlineCode = (cleanContent.match(/`[^`]+`/g) || []).length;
  const totalCodeBlocks = codeBlocks + Math.ceil(inlineCode / 3); // Group inline code

  const mathEquations =
    (cleanContent.match(/\$\$[\s\S]*?\$\$/g) || []).length + // Block math
    (cleanContent.match(/\$[^$]+\$/g) || []).length; // Inline math

  const images = (cleanContent.match(/!\[.*?\]\(.*?\)/g) || []).length;

  const tables =
    (cleanContent.match(/\|.*?\|/g) || []).length > 0
      ? cleanContent.split("\n").filter((line) => line.includes("|")).length / 3
      : 0;

  const headings = (cleanContent.match(/^#{1,6}\s+/gm) || []).length;
  const lists =
    (cleanContent.match(/^[-*+]\s+/gm) || []).length +
    (cleanContent.match(/^\d+\.\s+/gm) || []).length;
  const links = (cleanContent.match(/\[.*?\]\(.*?\)/g) || []).length;

  // Determine complexity based on content analysis
  let complexity: "beginner" | "intermediate" | "advanced" = "intermediate";

  const complexityScore =
    totalCodeBlocks * 2 +
    mathEquations * 3 +
    (wordCount > 2000 ? 2 : 0) +
    tables * 1.5 +
    (headings > 10 ? 1 : 0);

  if (complexityScore <= 3) complexity = "beginner";
  else if (complexityScore >= 10) complexity = "advanced";

  return {
    wordCount,
    codeBlocks: totalCodeBlocks,
    mathEquations,
    images,
    tables: Math.ceil(tables),
    headings,
    lists,
    links,
    complexity,
  };
}

/**
 * Calculates read time based on content analysis
 */
export function calculateReadTime(
  content: string,
  config: ReadTimeConfig = {}
): ReadTimeResult {
  const finalConfig = { ...DEFAULT_CONFIG, ...config };
  const analysis = analyzeContent(content);

  // Base reading time in minutes
  const baseReadingTime = analysis.wordCount / finalConfig.wordsPerMinute;

  // Additional time for different content types
  const codeBlockTime = (analysis.codeBlocks * finalConfig.codeBlockTime) / 60;
  const mathTime = (analysis.mathEquations * finalConfig.mathEquationTime) / 60;
  const imageTime = (analysis.images * finalConfig.imageTime) / 60;
  const tableTime = (analysis.tables * finalConfig.tableTime) / 60;

  // Complexity adjustment
  let complexityMultiplier = 1.0;
  switch (analysis.complexity) {
    case "beginner":
      complexityMultiplier = 0.9;
      break;
    case "advanced":
      complexityMultiplier = 1.2;
      break;
    default:
      complexityMultiplier = 1.0;
  }

  const complexityAdjustment =
    baseReadingTime * complexityMultiplier - baseReadingTime;

  // Calculate total time
  let totalTime =
    baseReadingTime +
    codeBlockTime +
    mathTime +
    imageTime +
    tableTime +
    complexityAdjustment;

  // Apply min/max constraints
  totalTime = Math.max(finalConfig.minReadTime, totalTime);
  totalTime = Math.min(finalConfig.maxReadTime, totalTime);

  // Round to nearest minute for clean display
  const readTimeMinutes = Math.ceil(totalTime);
  const readTime = `${readTimeMinutes} min read`;

  return {
    readTime,
    readTimeMinutes,
    analysis,
    breakdown: {
      baseReadingTime,
      codeBlockTime,
      mathTime,
      imageTime,
      tableTime,
      complexityAdjustment,
      totalTime,
    },
  };
}

/**
 * Quick function to get just the read time string (most common use case)
 */
export function getReadTime(content: string, config?: ReadTimeConfig): string {
  return calculateReadTime(content, config).readTime;
}

/**
 * Function to calculate read time from frontmatter tags for enhanced accuracy
 */
export function calculateReadTimeWithTags(
  content: string,
  tags: string[] = [],
  category: string = "",
  config: ReadTimeConfig = {}
): ReadTimeResult {
  const baseResult = calculateReadTime(content, config);

  // Adjust based on technical tags
  const technicalTags = [
    "machine-learning",
    "deep-learning",
    "ai",
    "algorithms",
    "mathematics",
    "typescript",
    "react",
    "next.js",
    "programming",
    "software-engineering",
    "blockchain",
    "cryptography",
    "defi",
    "smart-contracts",
    "web3",
    "system-design",
    "architecture",
    "microservices",
    "devops",
    "research",
    "paper-review",
    "academic",
  ];

  const hasTechnicalContent =
    tags.some((tag) =>
      technicalTags.some(
        (techTag) =>
          tag.toLowerCase().includes(techTag) ||
          techTag.includes(tag.toLowerCase())
      )
    ) ||
    technicalTags.some((techTag) => category.toLowerCase().includes(techTag));

  if (hasTechnicalContent && baseResult.analysis.complexity !== "advanced") {
    // Bump complexity up for technical content
    const adjustedTime = baseResult.readTimeMinutes * 1.1;
    const finalTime = Math.ceil(
      Math.max(baseResult.readTimeMinutes, adjustedTime)
    );

    return {
      ...baseResult,
      readTime: `${finalTime} min read`,
      readTimeMinutes: finalTime,
      breakdown: {
        ...baseResult.breakdown,
        complexityAdjustment:
          baseResult.breakdown.complexityAdjustment +
          (finalTime - baseResult.readTimeMinutes),
        totalTime: finalTime,
      },
    };
  }

  return baseResult;
}
