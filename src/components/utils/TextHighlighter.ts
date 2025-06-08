/**
 * Advanced Text Highlighter for Audio Reading
 * Handles text highlighting with paragraph-level and word-level granularity
 * Excludes code blocks and mathematical formulas from reading
 */

export interface WordInfo {
  element: HTMLElement;
  text: string;
  startIndex: number;
  endIndex: number;
  paragraphIndex: number;
}

export interface ParagraphInfo {
  element: HTMLElement;
  text: string;
  startIndex: number;
  endIndex: number;
  wordElements: HTMLElement[];
  lastWordIndex: number; // Index of the last word in this paragraph
}

export interface HighlightColors {
  wordHighlight: string;
  paragraphHighlight: string;
}

export class TextHighlighter {
  private container: HTMLElement;
  private words: WordInfo[] = [];
  private paragraphs: ParagraphInfo[] = [];
  private colors: HighlightColors;
  private currentWordElement: HTMLElement | null = null;
  private currentParagraphElement: HTMLElement | null = null;
  private currentParagraphIndex: number = -1;
  private originalHTML: string = "";
  private readableText: string = "";

  // CSS classes for highlighting
  private readonly WORD_HIGHLIGHT_CLASS = "highlighted-word";
  private readonly PARAGRAPH_HIGHLIGHT_CLASS = "highlighted-paragraph";

  constructor(container: HTMLElement, colors: HighlightColors) {
    this.container = container;
    this.colors = colors;
    this.originalHTML = container.innerHTML;
    this.processContent();
  }

  /**
   * Process content and prepare it for highlighting
   */
  private processContent(): void {
    this.words = [];
    this.paragraphs = [];

    // First, wrap words in spans for highlighting
    this.wrapWordsInSpans();

    // Then build word and paragraph mappings
    this.buildWordAndParagraphMappings();

    // Generate readable text
    this.generateReadableText();
  }

  /**
   * Wrap individual words in spans for precise highlighting
   */
  private wrapWordsInSpans(): void {
    const walker = document.createTreeWalker(
      this.container,
      NodeFilter.SHOW_TEXT,
      {
        acceptNode: (node) => {
          const parent = node.parentElement;
          if (!parent) return NodeFilter.FILTER_REJECT;

          // Skip elements that shouldn't be read
          if (this.shouldSkipElement(parent)) {
            return NodeFilter.FILTER_REJECT;
          }

          const text = node.textContent?.trim();
          return text && text.length > 0
            ? NodeFilter.FILTER_ACCEPT
            : NodeFilter.FILTER_REJECT;
        },
      }
    );

    const textNodes: Text[] = [];
    let node;
    while ((node = walker.nextNode())) {
      textNodes.push(node as Text);
    }

    // Process text nodes and wrap words in spans
    textNodes.forEach((textNode) => {
      const text = textNode.textContent || "";
      const parent = textNode.parentNode;

      if (!parent || text.trim().length === 0) return;

      // Split text into words and spaces, preserving whitespace
      const parts = text.split(/(\s+)/);
      const fragment = document.createDocumentFragment();

      parts.forEach((part) => {
        if (/\s+/.test(part)) {
          // Preserve whitespace
          fragment.appendChild(document.createTextNode(part));
        } else if (part.trim().length > 0) {
          // Wrap word in span
          const span = document.createElement("span");
          span.className = "word-span";
          span.textContent = part;
          fragment.appendChild(span);
        }
      });

      parent.replaceChild(fragment, textNode);
    });
  }

  /**
   * Build mappings of words and paragraphs
   */
  private buildWordAndParagraphMappings(): void {
    let characterIndex = 0;
    let paragraphIndex = 0;

    // Find all paragraph elements, prioritizing headings and including lists for better highlighting
    // We use a comprehensive selector to ensure we catch all relevant text containers
    const paragraphElements = this.container.querySelectorAll(
      "h1, h2, h3, h4, h5, h6, p, div, article, section, li, ul, ol"
    );

    paragraphElements.forEach((paragraphEl) => {
      const htmlElement = paragraphEl as HTMLElement;

      // Skip if this paragraph should be excluded
      if (this.shouldSkipElement(htmlElement)) return;

      // For div elements, only include them if they contain direct text content
      // and don't contain other paragraph-level elements
      if (htmlElement.tagName.toLowerCase() === "div") {
        const hasNestedParagraphs = htmlElement.querySelector(
          "h1, h2, h3, h4, h5, h6, p, article, section"
        );
        if (hasNestedParagraphs) return;
      }

      // For ul/ol elements, skip them as containers but process their li children
      if (
        htmlElement.tagName.toLowerCase() === "ul" ||
        htmlElement.tagName.toLowerCase() === "ol"
      ) {
        return; // Skip container, li elements will be processed separately
      }

      const wordElements = htmlElement.querySelectorAll(".word-span");
      if (wordElements.length === 0) return;

      const paragraphStartIndex = characterIndex;
      const paragraphWords: HTMLElement[] = [];
      let paragraphText = "";

      // Process words in this paragraph
      wordElements.forEach((wordEl) => {
        const wordElement = wordEl as HTMLElement;
        const wordText = wordElement.textContent || "";

        if (wordText.trim().length === 0) return;

        // Add word info
        const wordInfo: WordInfo = {
          element: wordElement,
          text: wordText,
          startIndex: characterIndex,
          endIndex: characterIndex + wordText.length - 1,
          paragraphIndex: paragraphIndex,
        };

        this.words.push(wordInfo);
        paragraphWords.push(wordElement);
        paragraphText += wordText + " ";
        characterIndex += wordText.length + 1; // +1 for space
      });

      // Add paragraph info
      const paragraphInfo: ParagraphInfo = {
        element: htmlElement,
        text: paragraphText.trim(),
        startIndex: paragraphStartIndex,
        endIndex: characterIndex - 1,
        wordElements: paragraphWords,
        lastWordIndex: this.words.length - 1,
      };

      this.paragraphs.push(paragraphInfo);
      paragraphIndex++;
    });
  }

  /**
   * Generate readable text from processed content
   */
  private generateReadableText(): void {
    let readableText = "";
    let currentParagraphIndex = -1;

    this.words.forEach((word) => {
      // Check if we're starting a new paragraph
      if (word.paragraphIndex !== currentParagraphIndex) {
        currentParagraphIndex = word.paragraphIndex;
        const paragraph = this.paragraphs[currentParagraphIndex];

        if (paragraph) {
          const tagName = paragraph.element.tagName.toLowerCase();

          // Add period after previous paragraph if it's a heading or list item for natural pause
          if (
            readableText.length > 0 &&
            (tagName.match(/^h[1-6]$/) || tagName === "li")
          ) {
            // Check if the previous text doesn't already end with punctuation
            const lastChar = readableText.trim().slice(-1);
            if (lastChar && !lastChar.match(/[.!?:;]/)) {
              readableText += ".";
            }
          }

          // Add some spacing between paragraphs for better reading flow
          if (readableText.length > 0) {
            readableText += " ";
          }
        }
      }

      readableText += word.text + " ";
    });

    this.readableText = readableText.trim();
  }

  /**
   * Check if an element should be skipped during reading
   */
  private shouldSkipElement(element: HTMLElement): boolean {
    if (!element) return true;

    const tagName = element.tagName.toLowerCase();
    const className = element.className || "";

    // Skip code blocks
    if (
      tagName === "code" ||
      tagName === "pre" ||
      className.includes("code") ||
      className.includes("highlight")
    ) {
      return true;
    }

    // Skip mathematical formulas (common patterns)
    if (
      className.includes("math") ||
      className.includes("formula") ||
      element.hasAttribute("data-math")
    ) {
      return true;
    }

    // Skip navigation elements
    if (tagName === "nav" || className.includes("nav")) {
      return true;
    }

    // Skip hidden elements
    if (
      element.style.display === "none" ||
      element.style.visibility === "hidden"
    ) {
      return true;
    }

    return false;
  }

  /**
   * Get readable text for speech synthesis
   */
  public getReadableText(): string {
    return this.readableText;
  }

  /**
   * Get word information by character index
   */
  public getWordByCharacterIndex(charIndex: number): WordInfo | null {
    return (
      this.words.find(
        (word) => charIndex >= word.startIndex && charIndex <= word.endIndex
      ) || null
    );
  }

  /**
   * Get paragraph information by character index
   */
  public getParagraphByCharacterIndex(charIndex: number): ParagraphInfo | null {
    return (
      this.paragraphs.find(
        (paragraph) =>
          charIndex >= paragraph.startIndex && charIndex <= paragraph.endIndex
      ) || null
    );
  }

  /**
   * Highlight word at the specified character index
   */
  public highlightWordAtIndex(charIndex: number): void {
    const wordInfo = this.getWordByCharacterIndex(charIndex);
    if (!wordInfo) return;

    // Clear previous word highlight
    this.clearWordHighlight();

    // Apply word highlight
    this.currentWordElement = wordInfo.element;
    this.currentWordElement.classList.add(this.WORD_HIGHLIGHT_CLASS);
    this.currentWordElement.style.backgroundColor = this.colors.wordHighlight;

    // Always update paragraph highlighting when word changes
    const paragraphInfo = this.paragraphs[wordInfo.paragraphIndex];
    if (paragraphInfo) {
      // If we're moving to a different paragraph, highlight the new one
      if (wordInfo.paragraphIndex !== this.currentParagraphIndex) {
        this.highlightParagraph(paragraphInfo);
        this.currentParagraphIndex = wordInfo.paragraphIndex;
      }
      // If we're still in the same paragraph but don't have it highlighted, highlight it
      else if (!this.currentParagraphElement) {
        this.highlightParagraph(paragraphInfo);
        this.currentParagraphIndex = wordInfo.paragraphIndex;
      }
    }
  }

  /**
   * Highlight paragraph with special handling for headings
   */
  private highlightParagraph(paragraphInfo: ParagraphInfo): void {
    // Clear previous paragraph highlight
    this.clearParagraphHighlight();

    // Apply paragraph highlight
    this.currentParagraphElement = paragraphInfo.element;
    this.currentParagraphElement.classList.add(this.PARAGRAPH_HIGHLIGHT_CLASS);

    // Apply different highlighting for headings vs regular paragraphs
    const tagName = this.currentParagraphElement.tagName.toLowerCase();
    if (tagName.match(/^h[1-6]$/)) {
      // For headings, use a more prominent highlight with special styling
      this.currentParagraphElement.classList.add("heading-highlight");
      this.currentParagraphElement.style.backgroundColor =
        this.colors.paragraphHighlight;
      this.currentParagraphElement.style.borderLeft =
        "4px solid " + this.colors.wordHighlight;
      this.currentParagraphElement.style.paddingLeft = "12px";
      this.currentParagraphElement.style.transition = "all 0.3s ease";
      this.currentParagraphElement.style.borderRadius = "4px";
      this.currentParagraphElement.style.marginLeft = "-8px";

      // Special styling for h2 elements
      if (tagName === "h2") {
        this.currentParagraphElement.style.boxShadow =
          "0 2px 8px rgba(0, 0, 0, 0.1)";
        this.currentParagraphElement.style.borderLeftWidth = "6px";
      }
    } else {
      // For regular paragraphs, use standard highlight
      this.currentParagraphElement.style.backgroundColor =
        this.colors.paragraphHighlight;
      this.currentParagraphElement.style.borderRadius = "4px";
      this.currentParagraphElement.style.padding = "4px 8px";
      this.currentParagraphElement.style.transition = "all 0.3s ease";
    }

    // Auto-scroll to the highlighted paragraph if enabled
    if (this.currentParagraphElement) {
      this.currentParagraphElement.scrollIntoView({
        behavior: "smooth",
        block: "center",
        inline: "nearest",
      });
    }
  }

  /**
   * Clear word highlighting
   */
  public clearWordHighlight(): void {
    if (this.currentWordElement) {
      this.currentWordElement.classList.remove(this.WORD_HIGHLIGHT_CLASS);
      this.currentWordElement.style.backgroundColor = "";
      this.currentWordElement = null;
    }
  }

  /**
   * Clear paragraph highlighting
   */
  public clearParagraphHighlight(): void {
    if (this.currentParagraphElement) {
      this.currentParagraphElement.classList.remove(
        this.PARAGRAPH_HIGHLIGHT_CLASS
      );
      this.currentParagraphElement.style.backgroundColor = "";

      // Clear additional heading styles if they were applied
      const tagName = this.currentParagraphElement.tagName.toLowerCase();
      if (tagName.match(/^h[1-6]$/)) {
        this.currentParagraphElement.classList.remove("heading-highlight");
        this.currentParagraphElement.style.borderLeft = "";
        this.currentParagraphElement.style.paddingLeft = "";
        this.currentParagraphElement.style.borderRadius = "";
        this.currentParagraphElement.style.marginLeft = "";

        // Clear special h2 styles
        if (tagName === "h2") {
          this.currentParagraphElement.style.boxShadow = "";
          this.currentParagraphElement.style.borderLeftWidth = "";
        }
      } else {
        // Clear regular paragraph styles
        this.currentParagraphElement.style.borderRadius = "";
        this.currentParagraphElement.style.padding = "";
      }

      this.currentParagraphElement.style.transition = "";
      this.currentParagraphElement = null;
    }
    this.currentParagraphIndex = -1;
  }

  /**
   * Clear all highlights
   */
  public clearAllHighlights(): void {
    this.clearWordHighlight();
    this.clearParagraphHighlight();
  }

  /**
   * Check if we've finished reading the current paragraph
   * This is called when a word is highlighted to determine if paragraph should remain highlighted
   */
  public shouldKeepParagraphHighlighted(charIndex: number): boolean {
    const wordInfo = this.getWordByCharacterIndex(charIndex);
    if (!wordInfo) return false;

    const paragraphInfo = this.paragraphs[wordInfo.paragraphIndex];
    if (!paragraphInfo) return false;

    // Find the current word index within all words
    const currentWordIndex = this.words.findIndex((w) => w === wordInfo);

    // Keep paragraph highlighted if this is not the last word in the paragraph
    return currentWordIndex < paragraphInfo.lastWordIndex;
  }

  /**
   * Handle end of speech - clear paragraph highlight only if we're at the last word
   */
  public onSpeechEnd(lastCharIndex: number): void {
    const wordInfo = this.getWordByCharacterIndex(lastCharIndex);
    if (!wordInfo) {
      this.clearAllHighlights();
      return;
    }

    const paragraphInfo = this.paragraphs[wordInfo.paragraphIndex];
    if (!paragraphInfo) {
      this.clearAllHighlights();
      return;
    }

    // Find the current word index within all words
    const currentWordIndex = this.words.findIndex((w) => w === wordInfo);

    // If this is the last word in the paragraph, clear paragraph highlight
    if (currentWordIndex === paragraphInfo.lastWordIndex) {
      this.clearParagraphHighlight();
    }

    // Always clear word highlight when speech ends
    this.clearWordHighlight();
  }

  /**
   * Scroll to the currently highlighted word
   */
  public scrollToCurrentWord(): void {
    if (this.currentWordElement) {
      this.currentWordElement.scrollIntoView({
        behavior: "smooth",
        block: "center",
        inline: "nearest",
      });
    }
  }

  /**
   * Get total number of words
   */
  public getWordCount(): number {
    return this.words.length;
  }

  /**
   * Get total number of paragraphs
   */
  public getParagraphCount(): number {
    return this.paragraphs.length;
  }

  /**
   * Get all words
   */
  public getWords(): WordInfo[] {
    return [...this.words];
  }

  /**
   * Get all paragraphs
   */
  public getParagraphs(): ParagraphInfo[] {
    return [...this.paragraphs];
  }

  /**
   * Reset to original content
   */
  public reset(): void {
    this.clearAllHighlights();
    this.container.innerHTML = this.originalHTML;
    this.words = [];
    this.paragraphs = [];
    this.currentWordElement = null;
    this.currentParagraphElement = null;
    this.currentParagraphIndex = -1;
    this.readableText = "";
  }

  /**
   * Cleanup method
   */
  public cleanup(): void {
    this.reset();
  }

  /**
   * Destroy method for complete cleanup
   */
  public destroy(): void {
    this.reset();
  }
}
