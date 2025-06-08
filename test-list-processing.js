// Test script to debug TextHighlighter list processing
function testListProcessing() {
  console.log("Starting list processing test...");

  // Navigate to blog post with lists
  if (
    window.location.pathname !==
    "/blog/software-development/algorithms-data-structures"
  ) {
    window.location.href =
      "/blog/software-development/algorithms-data-structures";
    return;
  }

  // Wait for page to load
  setTimeout(() => {
    const blogContent = document.querySelector(".prose");
    if (!blogContent) {
      console.error("Blog content not found");
      return;
    }

    console.log("Blog content found:", blogContent);

    // Check for list elements
    const lists = blogContent.querySelectorAll("ul, ol");
    const listItems = blogContent.querySelectorAll("li");

    console.log("Found lists:", lists.length);
    console.log("Found list items:", listItems.length);

    // Check list content
    listItems.forEach((li, index) => {
      console.log(`List item ${index + 1}:`, li.textContent?.trim());
    });

    // Test TextHighlighter if available
    if (window.TextHighlighter) {
      try {
        const highlighter = new window.TextHighlighter(blogContent, {
          wordHighlight: "#ffeb3b",
          paragraphHighlight: "#fff3c4",
        });

        const readableText = highlighter.getReadableText();
        console.log("Generated readable text length:", readableText.length);
        console.log("Readable text preview:", readableText.substring(0, 500));

        // Check if list content is included
        const hasListContent =
          /Choose the right data structure|Analyze complexity|Optimize for your use case/.test(
            readableText
          );
        console.log("Contains list content:", hasListContent);
      } catch (error) {
        console.error("Error testing TextHighlighter:", error);
      }
    } else {
      console.log("TextHighlighter not available in window object");
    }
  }, 2000);
}

// Run test
testListProcessing();
