# Automatic Read Time Calculation System

This project includes a comprehensive and intelligent read time calculation system that automatically estimates reading time for blog posts based on various content factors.

## Features

### 🧠 Intelligent Analysis

- **Word Count**: Accurate text parsing excluding code, math, and formatting
- **Content Type Detection**: Code blocks, math equations, images, tables
- **Complexity Assessment**: Beginner, intermediate, or advanced based on content
- **Technical Content Recognition**: Adjustments for technical blogs and tutorials

### ⚡ Performance Optimized

- **Fast Processing**: Efficient regex-based content parsing
- **Minimal Dependencies**: Uses only essential libraries
- **TypeScript Support**: Full type safety and IntelliSense

### 🎯 Accurate Estimates

- **Configurable WPM**: Default 200 words per minute, customizable
- **Content-Specific Adjustments**: Extra time for code, math, images
- **Complexity Multipliers**: Automatic adjustments based on content difficulty
- **Technical Tag Recognition**: Enhanced accuracy for technical content

## Quick Start

### Update All Blog Posts

```bash
npm run update-readtime
```

### Generate Analytics Report

```bash
npm run blog-report
```

### Validate Read Times

```bash
npm run blog-validate
```

## Usage Examples

### 1. Automatic Batch Update

```typescript
// Run this to update all blog posts at once
import { updateReadTimes } from "./scripts/updateReadTime";
await updateReadTimes();
```

### 2. Calculate for Single Post

```typescript
import { calculateReadTimeWithTags } from "./src/lib/readTimeCalculator";

const content = `# Your blog content here...`;
const tags = ["React", "TypeScript", "Tutorial"];
const category = "Programming";

const result = calculateReadTimeWithTags(content, tags, category);
console.log(result.readTime); // "8 min read"
```

### 3. React Hook Usage

```typescript
import { useReadTime } from "./src/components/hooks/useReadTime";

function BlogPreview({ content, tags, category }) {
  const { readTime, analysis } = useReadTime({
    content,
    tags,
    category,
  });

  return (
    <div>
      <span>{readTime}</span>
      <span>{analysis.wordCount} words</span>
    </div>
  );
}
```

### 4. Custom Configuration

```typescript
import { calculateReadTime } from "./src/lib/readTimeCalculator";

const customConfig = {
  wordsPerMinute: 180, // Slower reading speed
  codeBlockTime: 45, // More time for code blocks
  mathEquationTime: 20, // More time for math
  minReadTime: 2, // Minimum 2 minutes
  maxReadTime: 45, // Maximum 45 minutes
};

const result = calculateReadTime(content, customConfig);
```

## Algorithm Details

### Content Analysis Process

1. **Frontmatter Removal**: Strips YAML frontmatter from markdown
2. **Content Categorization**: Separates text, code, math, images, tables
3. **Word Counting**: Accurate text-only word count excluding markup
4. **Complexity Assessment**: Analyzes technical content density
5. **Time Calculation**: Applies configurable reading speeds and adjustments

### Reading Time Formula

```
Total Time = Base Reading Time + Additional Content Time + Complexity Adjustment

Where:
- Base Reading Time = Word Count ÷ Words Per Minute
- Additional Content Time = (Code Blocks × 30s) + (Math Equations × 15s) + (Images × 12s) + (Tables × 20s)
- Complexity Adjustment = Base Time × Complexity Multiplier (0.9 for beginner, 1.0 for intermediate, 1.2 for advanced)
```

### Complexity Detection

The system automatically determines content complexity based on:

- Number and length of code blocks
- Mathematical equations and formulas
- Content length and structure
- Technical vocabulary and concepts
- Table complexity and data density

## Content Type Recognition

### Code Blocks

- Fenced code blocks (```)
- Inline code (`code`)
- Technical content indicators

### Mathematical Content

- LaTeX block equations ($$...$$)
- Inline math expressions ($...$)
- Mathematical symbols and notation

### Media Content

- Images with alt text
- Complex tables and data
- Lists and structured content

### Technical Indicators

- Programming language tags
- Framework and library names
- Technical categories and tags

## Configuration Options

| Option             | Default | Description                       |
| ------------------ | ------- | --------------------------------- |
| `wordsPerMinute`   | 200     | Average reading speed             |
| `codeBlockTime`    | 30s     | Additional time per code block    |
| `mathEquationTime` | 15s     | Additional time per math equation |
| `imageTime`        | 12s     | Additional time per image         |
| `tableTime`        | 20s     | Additional time per table         |
| `minReadTime`      | 1 min   | Minimum read time                 |
| `maxReadTime`      | 60 min  | Maximum read time                 |

## Integration with Blog System

### Frontmatter Integration

The system automatically updates the `readTime` field in your markdown frontmatter:

```yaml
---
title: "Your Blog Post"
readTime: "8 min read" # Automatically calculated
tags: ["React", "TypeScript"]
category: "Tutorial"
---
```

### Build Process Integration

Add to your build process to ensure all posts have updated read times:

```json
{
  "scripts": {
    "prebuild": "npm run update-readtime",
    "build": "next build"
  }
}
```

## Analytics and Reporting

### Blog Analytics

Get comprehensive insights about your blog content:

```bash
npm run blog-report
```

This generates:

- Total posts and word count
- Average read time across all posts
- Complexity distribution analysis
- Detailed per-post breakdowns

### Validation

Ensure all read times are accurate and up-to-date:

```bash
npm run blog-validate
```

## Best Practices

### 1. Regular Updates

Run `npm run update-readtime` after:

- Adding new blog posts
- Editing existing content significantly
- Changing categories or tags

### 2. Content Optimization

Use the analytics to:

- Balance content complexity across your blog
- Optimize post length for target read times
- Identify outliers that may need review

### 3. Custom Configurations

Adjust settings based on your audience:

- Lower WPM for technical content
- Higher code block time for tutorials
- Custom complexity thresholds

## Technical Implementation

The read time calculator uses a sophisticated multi-step analysis process:

1. **Preprocessing**: Cleans and normalizes markdown content
2. **Tokenization**: Separates different content types using regex patterns
3. **Analysis**: Counts and categorizes content elements
4. **Calculation**: Applies reading speed and time adjustments
5. **Optimization**: Applies min/max constraints and rounding

The system is designed to be:

- **Accurate**: Handles complex markdown with embedded code and math
- **Fast**: Efficient processing suitable for build-time operations
- **Flexible**: Highly configurable for different content types
- **Maintainable**: Clean TypeScript code with comprehensive typing

## Contributing

To extend the read time calculator:

1. **Add new content types**: Extend the regex patterns in `analyzeContent()`
2. **Adjust complexity detection**: Modify the scoring algorithm
3. **Custom time calculations**: Add new adjustment factors
4. **Language support**: Add multi-language reading speed configs

The system is modular and extensible, making it easy to adapt for specific needs.
