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
  // Rules to avoid matching currency like "$5 and $10" while still catching
  // legitimate math that starts with a digit (e.g. "$320 \text{ KB}$"):
  //   - opening `$` is not preceded by `\` or `$`
  //   - opening `$` is not followed by whitespace
  //   - closing `$` is not preceded by whitespace
  //   - closing `$` is not followed by an alphanumeric (avoids "$5 and $10")
  //   - if content starts with a digit, it must also contain a LaTeX-ish
  //     character (\, {, }, ^, _) to distinguish math like "$320 \text{KB}$"
  //     from currency like "$5 and $10"
  //   - content allows escaped `\$`
  protectedContent = protectedContent.replace(
    /(?<![\\$])\$(?!\s)(?:(?=\d)(?=[^$\n]*[\\{}^_])|(?=\D))((?:[^$\n\\]|\\.)+?)(?<!\s)\$(?![A-Za-z0-9])/g,
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
