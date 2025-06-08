// Debug script to test TextHighlighter functionality
// Open browser console and run this in the algorithms blog post page

function debugTextHighlighter() {
  console.log("=== TextHighlighter Debug ===");

  // Find the blog content container
  const container =
    document.querySelector(".blog-content") ||
    document.querySelector("article") ||
    document.querySelector("main");

  if (!container) {
    console.error("Blog content container not found");
    return;
  }

  console.log("Container found:", container);

  // Look for list elements
  const lists = container.querySelectorAll("ul, ol");
  console.log(`Found ${lists.length} lists:`, lists);

  const listItems = container.querySelectorAll("li");
  console.log(`Found ${listItems.length} list items:`, listItems);

  // Show text content of list items
  listItems.forEach((li, index) => {
    console.log(`List item ${index}:`, li.textContent?.trim());
  });

  // Check if .word-span elements exist
  const wordSpans = container.querySelectorAll(".word-span");
  console.log(`Found ${wordSpans.length} word spans`);

  // Check paragraph elements that would be processed
  const paragraphElements = container.querySelectorAll(
    "h1, h2, h3, h4, h5, h6, p, div, article, section, li, ul, ol"
  );
  console.log(
    `Found ${paragraphElements.length} paragraph elements:`,
    paragraphElements
  );

  // Check if audio reader exists
  const audioButton =
    document.querySelector('[class*="audio"]') ||
    document.querySelector('button[title*="audio"]') ||
    document.querySelector('button[aria-label*="audio"]');
  console.log("Audio button found:", audioButton);

  return {
    container,
    lists: lists.length,
    listItems: listItems.length,
    wordSpans: wordSpans.length,
    paragraphElements: paragraphElements.length,
    audioButton,
  };
}

// Export for console use
window.debugTextHighlighter = debugTextHighlighter;

console.log("Debug function loaded. Run debugTextHighlighter() in console.");
