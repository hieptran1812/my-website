// Protect math blocks ($..$ and $$..$$) from being mangled by remark's markdown processing.
// Returns the modified text and the extracted blocks for later restoration.
export function protectMathBlocks(content: string): {
  protectedContent: string;
  mathBlocks: string[];
} {
  const mathBlocks: string[] = [];

  // Protect display math ($$...$$) first
  let protectedContent = content.replace(
    /\$\$([\s\S]*?)\$\$/g,
    (match: string) => {
      mathBlocks.push(match);
      return `<!--MATH_BLOCK_${mathBlocks.length - 1}-->`;
    },
  );

  // Then protect inline math ($...$).
  // Rules (Pandoc-style) to avoid matching currency like "$5 and $10":
  //   - opening `$` is not preceded by `\` or `$`
  //   - opening `$` is not followed by whitespace or a digit
  //   - closing `$` is not preceded by whitespace
  //   - closing `$` is not followed by a digit
  //   - content allows escaped `\$`
  protectedContent = protectedContent.replace(
    /(?<![\\$])\$(?![\s\d])((?:[^$\n\\]|\\.)+?)(?<!\s)\$(?!\d)/g,
    (match: string) => {
      mathBlocks.push(match);
      return `<!--MATH_BLOCK_${mathBlocks.length - 1}-->`;
    },
  );

  return { protectedContent, mathBlocks };
}

// Restore math blocks previously extracted by protectMathBlocks.
export function restoreMathBlocks(
  html: string,
  mathBlocks: string[],
): string {
  return html.replace(
    /<!--MATH_BLOCK_(\d+)-->/g,
    (_, index: string) => mathBlocks[Number(index)],
  );
}
