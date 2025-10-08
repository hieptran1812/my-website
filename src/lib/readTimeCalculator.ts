/**
 * Optimized Read Time Calculator
 * Calculates reading time based on content analysis and complexity factors
 */

// Cache for previously calculated read times to avoid redundant processing
const readTimeCache = new Map<string, ReadTimeResult>();

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

export interface ReadTimeBreakdown {
  baseReadingTime: number;
  codeBlockTime: number;
  mathTime: number;
  imageTime: number;
  tableTime: number;
  complexityAdjustment: number;
  totalTime: number;
}

export interface ReadTimeResult {
  readTime: string;
  readTimeMinutes: number;
  analysis: ContentAnalysis;
  breakdown: ReadTimeBreakdown;
}

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

const DEFAULT_CONFIG: Required<ReadTimeConfig> = {
  wordsPerMinute: 200,
  codeBlockTime: 30,
  mathEquationTime: 15,
  imageTime: 12,
  tableTime: 20,
  minReadTime: 1,
  maxReadTime: 60,
};

// Compile regular expressions once for better performance
const REGEX = {
  // Pre-compiled regex patterns for better performance
  frontMatter: /^---[\s\S]*?---\n/,
  codeBlocks: /```[\s\S]*?```/g,
  inlineCode: /`[^`]+`/g,
  images: /!\[.*?\]\(.*?\)/g,
  links: /\[.*?\]\(.*?\)/g,
  headings: /^#{1,6}\s+/gm,
  lists: /^[-*+]\s+/gm,
  numberedLists: /^\d+\.\s+/gm,
  tableSyntax: /\|.*?\|/g,
  blockMath: /\$\$[\s\S]*?\$\$/g,
  inlineMath: /\$[^$]+\$/g,
  whitespace: /\s+/g,
};

// Analyzes markdown content to extract various content types
export function analyzeContent(content: string): ContentAnalysis {
  // Remove frontmatter if present
  const cleanContent = content.replace(REGEX.frontMatter, "");

  // Count words (excluding code blocks and other special content)
  const textContent = cleanContent
    .replace(REGEX.codeBlocks, "") // Remove code blocks
    .replace(REGEX.inlineCode, "") // Remove inline code
    .replace(REGEX.images, "") // Remove images
    .replace(REGEX.links, "") // Remove links (keep text only)
    .replace(REGEX.headings, "") // Remove heading markers
    .replace(/[*_]{1,2}(.*?)[*_]{1,2}/g, "$1") // Remove emphasis markers
    .replace(REGEX.lists, "") // Remove list markers
    .replace(REGEX.numberedLists, "") // Remove numbered list markers
    .replace(REGEX.tableSyntax, "") // Remove table syntax
    .replace(REGEX.blockMath, "") // Remove block math
    .replace(REGEX.inlineMath, "") // Remove inline math
    .replace(REGEX.whitespace, " ") // Normalize whitespace
    .trim();

  const wordCount = textContent
    .split(/\s+/)
    .filter((word) => word.length > 0).length;

  // Count different content types
  const codeBlocks = (cleanContent.match(REGEX.codeBlocks) || []).length;
  const inlineCode = (cleanContent.match(REGEX.inlineCode) || []).length;
  const totalCodeBlocks = codeBlocks + Math.ceil(inlineCode / 3); // Group inline code

  const mathEquations =
    (cleanContent.match(REGEX.blockMath) || []).length + // Block math
    (cleanContent.match(REGEX.inlineMath) || []).length; // Inline math

  const images = (cleanContent.match(REGEX.images) || []).length;

  const tables =
    (cleanContent.match(REGEX.tableSyntax) || []).length > 0
      ? cleanContent.split("\n").filter((line) => line.includes("|")).length / 3
      : 0;

  const headings = (cleanContent.match(REGEX.headings) || []).length;
  const lists =
    (cleanContent.match(REGEX.lists) || []).length +
    (cleanContent.match(REGEX.numberedLists) || []).length;
  const links = (cleanContent.match(REGEX.links) || []).length;

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

  // Check cache first
  const cacheKey = JSON.stringify({ content, config: finalConfig });
  if (readTimeCache.has(cacheKey)) {
    return readTimeCache.get(cacheKey)!;
  }

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

  const result = {
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

  // Store in cache
  readTimeCache.set(cacheKey, result);

  return result;
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
  const technicalTags = ["machine learning"];

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
