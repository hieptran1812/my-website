/**
 * Advanced Speech Reader with Text Highlighting
 * Provides synchronized text highlighting with speech synthesis
 */

import { TextHighlighter, HighlightColors } from "./TextHighlighter";

export interface SpeechReaderOptions {
  highlightColors: HighlightColors;
  wordsPerMinute?: number;
  autoScroll?: boolean;
  voice?: SpeechSynthesisVoice | null;
  pitch?: number;
  rate?: number;
  volume?: number;
}

export interface SpeechReaderEvents {
  onStart?: () => void;
  onEnd?: () => void;
  onPause?: () => void;
  onResume?: () => void;
  onError?: (error: Error) => void;
  onProgress?: (progress: number) => void;
  onWordHighlight?: (word: string, position: number) => void;
  onSeek?: (position: number) => void;
}

export class SpeechReader {
  private container: HTMLElement;
  private highlighter: TextHighlighter;
  private utterance: SpeechSynthesisUtterance | null = null;
  private isPlaying = false;
  private isPaused = false;
  private currentText = "";
  private options: SpeechReaderOptions;
  private events: SpeechReaderEvents;
  private progressTimer: number | null = null;
  private startTime = 0;
  private pausedTime = 0;
  private totalPausedDuration = 0;
  private lastHeadingEndIndex = -1;
  private isWaitingAfterHeading = false;
  private currentCharIndex = 0;
  private lastCharIndexUpdate = 0;
  private estimatedWPM = 0;
  private actualSpeechStartTime = 0;
  private textSegments: {
    text: string;
    startIndex: number;
    endIndex: number;
  }[] = [];
  private currentSegmentIndex = 0;
  private highlightSyncTimer: number | null = null;

  // Enhanced precision tracking for seeking synchronization
  private precisionTracking = {
    baselineStartTime: 0,
    baselineCharIndex: 0,
    seekOffset: 0,
    lastSyncPoint: 0,
    charIndexHistory: [] as Array<{ charIndex: number; timestamp: number }>,
    realTimeWPM: 0,
    averageCharPerSecond: 0,
  };

  // Word-to-character mapping for precise positioning
  // Maps word index to character ranges in PROCESSED text
  private wordMappings: Array<{
    wordIndex: number;
    charStartProcessed: number; // Start index in processed text
    charEndProcessed: number; // End index in processed text
    charStartOriginal: number; // Start index in original text (for TextHighlighter)
    charEndOriginal: number; // End index in original text
    wordText: string;
  }> = [];

  // Total word count for progress calculation
  private totalWordCount: number = 0;

  // Seeking state management
  private seekingState = {
    isInSeekMode: false,
    targetPercentage: 0,
    targetCharIndex: 0,
    seekStartTime: 0,
    preSeekCharIndex: 0,
  };

  // Mapping between processed text indices and original text indices
  private processedToOriginalMap: Map<number, number> = new Map();
  private originalToProcessedMap: Map<number, number> = new Map();
  private originalText: string = "";

  constructor(
    container: HTMLElement,
    options: SpeechReaderOptions,
    events: SpeechReaderEvents = {}
  ) {
    // Check for Web Speech API support
    if (!("speechSynthesis" in window)) {
      throw new Error("Web Speech API is not supported in this browser");
    }

    this.container = container;
    this.options = {
      wordsPerMinute: 200,
      autoScroll: false,
      pitch: 1.0,
      rate: 0.85,
      volume: 1,
      ...options,
    };
    this.events = events;

    this.highlighter = new TextHighlighter(container, options.highlightColors);
    this.setupSpeechSynthesis();
  }

  /**
   * Split text into segments for seeking capability
   */
  private splitTextIntoSegments(): void {
    // Clear any existing segments first
    this.textSegments = [];

    if (!this.currentText || this.currentText.trim() === "") {
      return;
    }

    // Enhanced sentence splitting with better punctuation handling
    const sentencePattern = /([.!?:;]+(?:\s*["'"'])?)\s+/g;
    const sentences: string[] = [];
    let lastIndex = 0;
    let match;

    // Extract sentences with their punctuation, including quotes
    while ((match = sentencePattern.exec(this.currentText)) !== null) {
      const sentence = this.currentText.slice(
        lastIndex,
        match.index + match[1].length
      );
      if (sentence.trim()) {
        sentences.push(sentence.trim());
      }
      lastIndex = match.index + match[0].length;
    }

    // Add any remaining text as the last sentence
    if (lastIndex < this.currentText.length) {
      const remaining = this.currentText.slice(lastIndex).trim();
      if (remaining) {
        sentences.push(remaining);
      }
    }

    // Also split long sentences by commas for better seeking
    const finalSentences: string[] = [];
    for (const sentence of sentences) {
      if (sentence.length > 200) {
        // Split long sentences
        const commaSplit = sentence.split(/,\s+/);
        if (commaSplit.length > 1) {
          let accumulated = "";
          for (let i = 0; i < commaSplit.length; i++) {
            accumulated += commaSplit[i];
            if (i < commaSplit.length - 1) accumulated += ",";

            // Add segment when we reach a good length or at the end
            if (accumulated.length > 100 || i === commaSplit.length - 1) {
              finalSentences.push(accumulated.trim());
              accumulated = "";
            }
          }
        } else {
          finalSentences.push(sentence);
        }
      } else {
        finalSentences.push(sentence);
      }
    }

    // Create segments with accurate positioning
    let currentIndex = 0;
    const createdSegments = new Set<string>(); // Prevent duplicates

    for (const sentence of finalSentences) {
      if (sentence.trim()) {
        // Find the exact position of this sentence in the original text
        const startIndex = this.currentText.indexOf(sentence, currentIndex);

        if (startIndex !== -1 && startIndex >= currentIndex) {
          const endIndex = startIndex + sentence.length;

          // Create a unique key to prevent duplicate segments
          const segmentKey = `${startIndex}-${endIndex}`;

          if (!createdSegments.has(segmentKey)) {
            this.textSegments.push({
              text: sentence,
              startIndex,
              endIndex,
            });

            createdSegments.add(segmentKey);

            // Move current index past this sentence
            currentIndex = endIndex;
          }
        }
      }
    }
  }

  /**
   * Setup speech synthesis with options
   */
  private setupSpeechSynthesis(): void {
    // Wait for voices to be loaded
    if (speechSynthesis.getVoices().length === 0) {
      speechSynthesis.addEventListener("voiceschanged", () => {
        this.configureSpeech();
      });
    } else {
      this.configureSpeech();
    }
  }

  /**
   * Configure speech synthesis settings
   */
  private configureSpeech(): void {
    if (this.utterance) {
      const voice = this.options.voice || this.getPreferredVoice();
      this.utterance.voice = voice;
      this.utterance.pitch = this.options.pitch || 1;
      this.utterance.rate = this.options.rate || 0.85; // Match constructor default
      this.utterance.volume = this.options.volume || 1;
    }
  }

  /**
   * Get preferred voice (prioritize English voices)
   */
  private getPreferredVoice(): SpeechSynthesisVoice | null {
    const voices = speechSynthesis.getVoices();

    const preferredVoices = [
      "Google US English",
      "Microsoft Aria Online (Natural) - English (United States)",
      "Microsoft Guy Online (Natural) - English (United States)",
      "Alex",
      "Samantha",
      "Google UK English Female",
      "Google UK English Male",
      "Microsoft David - English (United States)",
      "Microsoft Zira - English (United States)",
    ];

    for (const voiceName of preferredVoices) {
      const voice = voices.find(
        (v) => v.name.includes(voiceName) || v.name === voiceName
      );
      if (voice && voice.lang.startsWith("en")) {
        return voice;
      }
    }

    const englishVoices = voices.filter((v) => v.lang.startsWith("en"));

    const onlineVoice = englishVoices.find(
      (v) =>
        v.name.toLowerCase().includes("online") ||
        v.name.toLowerCase().includes("neural") ||
        v.name.toLowerCase().includes("enhanced")
    );

    if (onlineVoice) {
      return onlineVoice;
    }

    return englishVoices[0] || voices[0] || null;
  }

  /**
   * Process text to add natural pauses for punctuation
   * Also builds a mapping between processed and original text indices
   */
  private processTextForNaturalSpeech(text: string): string {
    if (!text) return "";

    // Store original text for mapping
    this.originalText = text;
    this.processedToOriginalMap.clear();
    this.originalToProcessedMap.clear();

    // Build character-by-character mapping while processing
    let processedText = "";
    let processedIndex = 0;

    // First normalize whitespace and track mapping
    const normalizedText = text.replace(/\s+/g, " ");

    // Build initial mapping for normalized text
    let normalizedIndex = 0;
    const normalizedToOriginal: Map<number, number> = new Map();

    for (let i = 0; i < text.length; i++) {
      if (/\s/.test(text[i])) {
        // Skip consecutive whitespace in original, map to single space in normalized
        if (i === 0 || !/\s/.test(text[i - 1])) {
          normalizedToOriginal.set(normalizedIndex, i);
          normalizedIndex++;
        }
      } else {
        normalizedToOriginal.set(normalizedIndex, i);
        normalizedIndex++;
      }
    }

    // Now process the normalized text and build final mapping
    let i = 0;
    while (i < normalizedText.length) {
      const char = normalizedText[i];
      const origCharIndex = normalizedToOriginal.get(i) ?? i;

      // Map current processed position to original
      this.processedToOriginalMap.set(processedIndex, origCharIndex);
      this.originalToProcessedMap.set(origCharIndex, processedIndex);

      processedText += char;
      processedIndex++;

      // Add pauses based on punctuation (these are extra chars that map to same original position)
      let extraSpaces = 0;

      // Check for sentence endings
      if (/[.!?]/.test(char)) {
        extraSpaces = 4; // 5 spaces total (1 already added)
      }
      // Check for colons and semicolons
      else if (/[;:]/.test(char)) {
        extraSpaces = 2; // 3 spaces total
      }
      // Check for commas and dashes
      else if (/[,—–-]/.test(char)) {
        extraSpaces = 1; // 2 spaces total
      }

      // Add extra spaces for pauses, mapping them to the same original position
      for (let j = 0; j < extraSpaces; j++) {
        this.processedToOriginalMap.set(processedIndex, origCharIndex);
        processedText += " ";
        processedIndex++;
      }

      i++;
    }

    return processedText.trim();
  }

  /**
   * Get word index from processed text character index
   * This is the key method for syncing speech position with highlighting
   */
  private getWordIndexFromProcessedCharIndex(processedCharIndex: number): number {
    if (this.wordMappings.length === 0) return 0;

    // Binary search for efficiency
    let left = 0;
    let right = this.wordMappings.length - 1;

    while (left <= right) {
      const mid = Math.floor((left + right) / 2);
      const mapping = this.wordMappings[mid];

      if (
        processedCharIndex >= mapping.charStartProcessed &&
        processedCharIndex <= mapping.charEndProcessed
      ) {
        return mid;
      } else if (processedCharIndex < mapping.charStartProcessed) {
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    }

    // If not found exactly, return the closest word
    // This handles cases where charIndex is in whitespace between words
    for (let i = 0; i < this.wordMappings.length; i++) {
      if (this.wordMappings[i].charStartProcessed > processedCharIndex) {
        return Math.max(0, i - 1);
      }
    }

    return this.wordMappings.length - 1;
  }

  /**
   * Start reading the text
   */
  public start(): void {
    if (this.isPlaying) {
      this.stop();
    }

    const rawText = this.highlighter.getReadableText();

    if (!rawText) {
      this.events.onError?.(new Error("No readable text found"));
      return;
    }

    const text = this.processTextForNaturalSpeech(rawText);

    // Complete state reset before starting new session
    this.currentText = "";
    this.textSegments = [];
    this.currentCharIndex = 0;
    this.currentSegmentIndex = 0;
    this.totalPausedDuration = 0;
    this.lastCharIndexUpdate = 0;
    this.estimatedWPM = 0;
    this.startTime = 0;
    this.pausedTime = 0;
    this.isWaitingAfterHeading = false;
    this.lastHeadingEndIndex = -1;

    this.currentText = text;
    this.splitTextIntoSegments();
    this.initializePrecisionTracking();

    this.utterance = new SpeechSynthesisUtterance(text);
    this.configureSpeech();

    this.utterance.onstart = () => {
      this.isPlaying = true;
      this.isPaused = false;
      this.startTime = Date.now();
      this.totalPausedDuration = 0;
      this.startProgressTracking();
      this.events.onStart?.();
    };

    this.utterance.onend = () => {
      this.isPlaying = false;
      this.isPaused = false;
      this.highlighter.clearAllHighlights();
      this.stopProgressTracking();
      this.events.onEnd?.();
    };

    this.utterance.onerror = (event: SpeechSynthesisErrorEvent) => {
      // Handle errors more gracefully - don't log common cancellation errors
      const error = event.error;
      if (error === "canceled" || error === "interrupted") {
        // These are common when speech is stopped/restarted, just reset state silently
        this.isPlaying = false;
        this.isPaused = false;
        this.highlighter.clearAllHighlights();
        this.stopProgressTracking();
        return;
      }

      this.isPlaying = false;
      this.isPaused = false;
      this.highlighter.clearAllHighlights();
      this.stopProgressTracking();
      this.events.onError?.(new Error(`Speech synthesis error: ${error}`));
    };

    this.utterance.onpause = () => {
      this.isPaused = true;
      this.pausedTime = Date.now();
      this.stopProgressTracking();
      this.events.onPause?.();
    };

    this.utterance.onresume = () => {
      this.isPaused = false;
      this.totalPausedDuration += Date.now() - this.pausedTime;
      this.startProgressTracking();
      this.events.onResume?.();
    };

    // Handle word boundary events for highlighting - Enhanced synchronization
    this.utterance.onboundary = (event) => {
      if (event.name === "word") {
        // Update current position with timing info
        this.updateCurrentPosition(event.charIndex);

        // Handle heading pause logic
        this.handleHeadingPause(event.charIndex);

        // Enhanced highlighting with better synchronization
        this.highlightWithSync(event.charIndex);

        this.events.onWordHighlight?.(
          this.getWordAtIndex(event.charIndex),
          event.charIndex
        );
      }
    };

    try {
      if (speechSynthesis.speaking || speechSynthesis.pending) {
        speechSynthesis.cancel();
        setTimeout(() => {
          speechSynthesis.speak(this.utterance!);
        }, 150);
      } else {
        speechSynthesis.speak(this.utterance);
      }
    } catch (error) {
      this.events.onError?.(new Error(`Failed to start speech: ${error}`));
    }
  }

  /**
   * Pause the speech
   */
  public pause(): void {
    if (this.isPlaying && !this.isPaused) {
      speechSynthesis.pause();
      // Update state immediately for responsive UI
      this.isPaused = true;
      this.pausedTime = Date.now();
      this.stopProgressTracking();
      this.events.onPause?.();
    }
  }

  /**
   * Resume the speech
   */
  public resume(): void {
    if (this.isPlaying && this.isPaused) {
      speechSynthesis.resume();
      // Update state immediately for responsive UI
      this.isPaused = false;
      this.totalPausedDuration += Date.now() - this.pausedTime;
      this.startProgressTracking();
      this.events.onResume?.();
    }
  }

  /**
   * Stop the speech
   */
  public stop(): void {
    // Cancel speech synthesis first
    speechSynthesis.cancel();

    // Reset all state variables immediately
    this.isPlaying = false;
    this.isPaused = false;
    this.currentCharIndex = 0;
    this.currentSegmentIndex = 0;
    this.totalPausedDuration = 0;
    this.lastCharIndexUpdate = 0;
    this.estimatedWPM = 0;
    this.startTime = 0;
    this.pausedTime = 0;
    this.isWaitingAfterHeading = false;
    this.lastHeadingEndIndex = -1;

    // Clear text-related state to prevent accumulation
    this.currentText = "";
    this.textSegments = [];
    this.originalText = "";
    this.processedToOriginalMap.clear();
    this.originalToProcessedMap.clear();

    // Clear all timers and highlighting
    this.stopProgressTracking();
    this.stopBackupHighlighting();
    this.highlighter.clearAllHighlights();

    // Clear utterance to prevent any residual events
    if (this.utterance) {
      this.utterance.onstart = null;
      this.utterance.onend = null;
      this.utterance.onerror = null;
      this.utterance.onpause = null;
      this.utterance.onresume = null;
      this.utterance.onboundary = null;
      this.utterance = null;
    }

  }

  /**
   * Seek to a specific position (0-100%) - Optimized for accurate sync
   * Uses word-based targeting for accurate synchronization
   */
  public seekTo(percentage: number): void {
    if (!this.currentText || this.wordMappings.length === 0) {
      return;
    }

    percentage = Math.max(0, Math.min(100, percentage));

    const targetWordIndex = Math.floor(
      (percentage / 100) * this.totalWordCount
    );
    const clampedWordIndex = Math.max(
      0,
      Math.min(targetWordIndex, this.wordMappings.length - 1)
    );

    const targetWordMapping = this.wordMappings[clampedWordIndex];
    const targetProcessedIndex = targetWordMapping
      ? targetWordMapping.charStartProcessed
      : 0;

    const targetSegmentIndex = this.findSegmentByCharIndex(targetProcessedIndex);

    if (targetSegmentIndex === -1) {
      return;
    }

    // Store current state for restoration
    const wasPlaying = this.isPlaying;
    const wasPaused = this.isPaused;
    const preservedText = this.currentText;
    const preservedSegments = [...this.textSegments];
    const preservedWordMappings = [...this.wordMappings];
    const preservedOriginalText = this.originalText;
    const preservedProcessedToOriginal = new Map(this.processedToOriginalMap);
    const preservedOriginalToProcessed = new Map(this.originalToProcessedMap);

    // Cancel current speech and reset state
    this.cancelCurrentSpeech();

    // Restore preserved data including mappings
    this.currentText = preservedText;
    this.textSegments = preservedSegments;
    this.wordMappings = preservedWordMappings;
    this.originalText = preservedOriginalText;
    this.processedToOriginalMap = preservedProcessedToOriginal;
    this.originalToProcessedMap = preservedOriginalToProcessed;

    // Update position state FIRST - Critical for sync
    this.updateSeekPosition(
      targetProcessedIndex,
      targetSegmentIndex,
      percentage
    );

    // Immediately update highlighting using ORIGINAL text index
    this.synchronizeHighlightingToPosition(targetProcessedIndex);

    // Force immediate progress update to prevent UI lag
    this.forceProgressUpdate(percentage);

    // Create new utterance from target position if we have remaining text
    const remainingText = this.getRemainingTextFromIndex(targetProcessedIndex);
    if (remainingText && remainingText.trim()) {
      this.createSeekUtterance(remainingText, targetProcessedIndex);

      // Resume playback if it was active before seeking
      if (wasPlaying && !wasPaused) {
        this.resumePlaybackAfterSeek();
      }
    }

    const actualPercentage = this.getCurrentProgress();
    this.events.onSeek?.(actualPercentage);
  }

  /**
   * Cancel current speech and clean up state (preserves text mappings for seeking)
   */
  private cancelCurrentSpeech(): void {
    speechSynthesis.cancel();

    this.isPlaying = false;
    this.isPaused = false;
    this.stopProgressTracking();
    this.stopBackupHighlighting();

    // Clear utterance events to prevent conflicts
    // Note: We don't clear text mappings here as they're needed for seeking
    if (this.utterance) {
      this.utterance.onstart = null;
      this.utterance.onend = null;
      this.utterance.onerror = null;
      this.utterance.onpause = null;
      this.utterance.onresume = null;
      this.utterance.onboundary = null;
      this.utterance = null;
    }
  }

  /**
   * Update position state after seeking
   */
  private updateSeekPosition(
    targetCharIndex: number,
    targetSegmentIndex: number,
    percentage: number
  ): void {
    // Update character and segment position
    this.currentCharIndex = targetCharIndex;
    this.currentSegmentIndex = targetSegmentIndex;
    this.lastCharIndexUpdate = Date.now();

    // Recalibrate timing to match the new position
    this.recalibrateTimingForSeek(percentage);

    // Reset precision tracking with new baseline
    this.resetPrecisionTrackingAfterSeek(targetCharIndex);
  }

  /**
   * Synchronize highlighting to the exact position
   * Uses word-based indexing for accurate synchronization
   */
  private synchronizeHighlightingToPosition(processedCharIndex: number): void {
    try {
      this.highlighter.clearAllHighlights();
      const wordIndex = this.getWordIndexFromProcessedCharIndex(processedCharIndex);
      this.highlighter.highlightWordByIndex(wordIndex);
    } catch {
      this.fallbackHighlighting(processedCharIndex);
    }
  }

  /**
   * Force immediate progress update to prevent UI lag
   */
  private forceProgressUpdate(targetPercentage: number): void {
    this.events.onProgress?.(targetPercentage);

    setTimeout(() => {
      this.events.onProgress?.(this.getCurrentProgress());
    }, 50);

    setTimeout(() => {
      if (this.isPlaying) {
        this.events.onProgress?.(this.getCurrentProgress());
      }
    }, 200);
  }

  /**
   * Create new utterance for seeking with optimized event handlers
   */
  private createSeekUtterance(
    remainingText: string,
    startCharIndex: number
  ): void {
    this.utterance = new SpeechSynthesisUtterance(remainingText);
    this.configureSpeech();

    // Use optimized event handlers for seeking
    this.setupOptimizedSeekEventHandlers(startCharIndex);
  }

  /**
   * Resume playback after seeking with error handling
   */
  private resumePlaybackAfterSeek(): void {
    if (!this.utterance) {
      return;
    }

    try {
      speechSynthesis.speak(this.utterance);
      this.isPlaying = true;
      this.isPaused = false;
      this.startProgressTracking();
      this.startBackupHighlighting();
    } catch (error) {
      this.events.onError?.(new Error(`Failed to resume speech: ${error}`));
    }
  }

  /**
   * Recalibrate timing specifically for seeking operations
   */
  private recalibrateTimingForSeek(targetPercentage: number): void {
    const totalDuration = this.estimateDuration();
    const targetElapsedTime = (targetPercentage / 100) * totalDuration;

    this.startTime = Date.now() - targetElapsedTime;
    this.totalPausedDuration = 0;
    this.estimatedWPM = 0;
  }

  /**
   * Reset precision tracking after seeking
   */
  private resetPrecisionTrackingAfterSeek(charIndex: number): void {
    const now = Date.now();
    this.precisionTracking = {
      baselineStartTime: now,
      baselineCharIndex: charIndex,
      seekOffset: 0,
      lastSyncPoint: now,
      charIndexHistory: [{ charIndex, timestamp: now }],
      realTimeWPM: this.options.wordsPerMinute || 200,
      averageCharPerSecond: 0,
    };
  }

  /**
   * Setup optimized event handlers for seek operations
   * @param startProcessedIndex - The starting index in PROCESSED text
   */
  private setupOptimizedSeekEventHandlers(startProcessedIndex: number): void {
    if (!this.utterance) return;

    this.utterance.onstart = () => {
      this.isPlaying = true;
      this.isPaused = false;
      this.startProgressTracking();
      this.startBackupHighlighting();
      this.events.onStart?.();
    };

    this.utterance.onend = () => {
      this.isPlaying = false;
      this.isPaused = false;
      this.highlighter.clearAllHighlights();
      this.stopProgressTracking();
      this.events.onEnd?.();
    };

    this.utterance.onerror = (event: SpeechSynthesisErrorEvent) => {
      const error = event.error;
      if (error === "canceled" || error === "interrupted") {
        this.isPlaying = false;
        this.isPaused = false;
        this.highlighter.clearAllHighlights();
        this.stopProgressTracking();
        return;
      }

      this.isPlaying = false;
      this.isPaused = false;
      this.highlighter.clearAllHighlights();
      this.stopProgressTracking();
      this.events.onError?.(new Error(`Speech synthesis error: ${error}`));
    };

    this.utterance.onpause = () => {
      this.isPaused = true;
      this.pausedTime = Date.now();
      this.stopProgressTracking();
      this.events.onPause?.();
    };

    this.utterance.onresume = () => {
      this.isPaused = false;
      this.totalPausedDuration += Date.now() - this.pausedTime;
      this.startProgressTracking();
      this.events.onResume?.();
    };

    this.utterance.onboundary = (event) => {
      if (event.name === "word") {
        // event.charIndex is relative to the remaining text (from seek position)
        const relativeCharIndex = event.charIndex;

        // Calculate actual index in PROCESSED text
        const actualProcessedIndex = startProcessedIndex + relativeCharIndex;

        if (this.validateSeekPosition(actualProcessedIndex)) {
          // Update position tracking with processed index
          this.updateCurrentPositionOptimized(actualProcessedIndex);

          // Use word-based highlighting for accuracy
          const wordIndex = this.getWordIndexFromProcessedCharIndex(actualProcessedIndex);
          this.highlighter.highlightWordByIndex(wordIndex);

          // Handle heading pauses
          this.handleHeadingPause(actualProcessedIndex);

          this.events.onWordHighlight?.(
            this.getWordAtIndex(actualProcessedIndex),
            actualProcessedIndex
          );
        } else {
          const estimatedProcessedIndex = Math.max(
            startProcessedIndex,
            this.estimateCurrentCharIndexFromTime()
          );
          if (this.validateSeekPosition(estimatedProcessedIndex)) {
            this.updateCurrentPositionOptimized(estimatedProcessedIndex);
            const wordIndex = this.getWordIndexFromProcessedCharIndex(estimatedProcessedIndex);
            this.highlighter.highlightWordByIndex(wordIndex);
          }
        }
      }
    };
  }

  /**
   * Optimized position update for better performance
   */
  private updateCurrentPositionOptimized(charIndex: number): void {
    this.currentCharIndex = charIndex;
    this.lastCharIndexUpdate = Date.now();

    // Optimized WPM calculation
    if (this.startTime && this.currentCharIndex > 0) {
      const elapsed =
        (Date.now() - this.startTime - this.totalPausedDuration) / 1000 / 60;
      const wordsSpoken = this.getWordCountUpToIndex(charIndex);
      if (elapsed > 0 && wordsSpoken > 0) {
        this.estimatedWPM = wordsSpoken / elapsed;
      }
    }

    // Update precision tracking efficiently
    this.updatePrecisionTrackingOptimized(charIndex);
  }

  /**
   * Optimized precision tracking update
   */
  private updatePrecisionTrackingOptimized(charIndex: number): void {
    const now = Date.now();

    // Add to history with size limit
    this.precisionTracking.charIndexHistory.push({
      charIndex,
      timestamp: now,
    });

    // Keep only last 5 entries for efficiency
    if (this.precisionTracking.charIndexHistory.length > 5) {
      this.precisionTracking.charIndexHistory.shift();
    }

    // Update metrics less frequently for performance
    if (this.precisionTracking.charIndexHistory.length >= 3) {
      this.calculateRealTimeMetricsOptimized();
    }

    this.precisionTracking.lastSyncPoint = now;
  }

  /**
   * Optimized real-time metrics calculation
   */
  private calculateRealTimeMetricsOptimized(): void {
    const history = this.precisionTracking.charIndexHistory;

    if (history.length < 2) return;

    const recent = history.slice(-2); // Use only last 2 points for efficiency
    const timeSpan = recent[1].timestamp - recent[0].timestamp;
    const charSpan = recent[1].charIndex - recent[0].charIndex;

    if (timeSpan > 0 && charSpan > 0) {
      this.precisionTracking.averageCharPerSecond =
        (charSpan / timeSpan) * 1000;

      // Simplified WPM estimation
      const avgCharsPerWord = 5; // Standard estimation
      this.precisionTracking.realTimeWPM =
        (this.precisionTracking.averageCharPerSecond * 60) / avgCharsPerWord;
    }
  }

  /**
   * Find segment index by character index - Optimized
   */
  private findSegmentByCharIndex(charIndex: number): number {
    // Binary search for better performance with large texts
    let left = 0;
    let right = this.textSegments.length - 1;

    while (left <= right) {
      const mid = Math.floor((left + right) / 2);
      const segment = this.textSegments[mid];

      if (charIndex >= segment.startIndex && charIndex <= segment.endIndex) {
        return mid;
      } else if (charIndex < segment.startIndex) {
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    }

    // Fallback to linear search if binary search fails
    for (let i = 0; i < this.textSegments.length; i++) {
      const segment = this.textSegments[i];
      if (charIndex >= segment.startIndex && charIndex <= segment.endIndex) {
        return i;
      }
    }

    return -1;
  }

  /**
   * Get remaining text from a specific character index
   */
  private getRemainingTextFromIndex(charIndex: number): string {
    if (charIndex >= this.currentText.length) {
      return "";
    }
    return this.currentText.substring(charIndex);
  }

  /**
   * Get total duration in seconds
   */
  public getTotalDuration(): number {
    return this.estimateDuration() / 1000;
  }

  /**
   * Get remaining time in seconds
   */
  public getRemainingTime(): number {
    if (!this.isPlaying) return this.getTotalDuration();

    const elapsed = Date.now() - this.startTime - this.totalPausedDuration;
    const estimatedDuration = this.estimateDuration();
    const remaining = Math.max(0, estimatedDuration - elapsed);
    return remaining / 1000;
  }

  /**
   * Update current position with enhanced tracking
   */
  private updateCurrentPosition(charIndex: number): void {
    this.currentCharIndex = charIndex;
    this.lastCharIndexUpdate = Date.now();

    // Calculate real-time WPM for better estimation
    if (this.startTime && this.currentCharIndex > 0) {
      const elapsed =
        (Date.now() - this.startTime - this.totalPausedDuration) / 1000 / 60; // minutes
      const wordsSpoken = this.getWordCountUpToIndex(charIndex);
      if (elapsed > 0 && wordsSpoken > 0) {
        this.estimatedWPM = wordsSpoken / elapsed;
      }
    }

    // Update precision tracking with current position
    this.updatePrecisionTracking(charIndex);
  }

  /**
   * Initialize precision tracking for enhanced synchronization
   */
  private initializePrecisionTracking(): void {
    this.buildWordMappings();
    this.resetPrecisionTracking();
  }

  /**
   * Build word-to-character mapping for precise seeking
   * Maps words to both processed text indices and original text indices
   */
  private buildWordMappings(): void {
    this.wordMappings = [];

    if (!this.currentText || !this.originalText) return;

    // Build mappings for PROCESSED text (used by speech synthesis)
    const processedWords = this.currentText.split(/(\s+)/);
    let processedCharPos = 0;

    // Build mappings for ORIGINAL text (used by TextHighlighter)
    const originalWords = this.originalText.split(/(\s+)/);
    let originalCharPos = 0;

    // Build original word positions first
    const originalWordPositions: Array<{ start: number; end: number }> = [];
    for (const word of originalWords) {
      if (word.trim()) {
        originalWordPositions.push({
          start: originalCharPos,
          end: originalCharPos + word.length - 1,
        });
      }
      originalCharPos += word.length;
    }

    // Now build processed word mappings with links to original positions
    let wordIndex = 0;
    for (const word of processedWords) {
      if (word.trim()) {
        const originalPos = originalWordPositions[wordIndex] || {
          start: 0,
          end: 0,
        };

        this.wordMappings.push({
          wordIndex,
          charStartProcessed: processedCharPos,
          charEndProcessed: processedCharPos + word.length - 1,
          charStartOriginal: originalPos.start,
          charEndOriginal: originalPos.end,
          wordText: word,
        });
        wordIndex++;
      }
      processedCharPos += word.length;
    }

    this.totalWordCount = this.wordMappings.length;
  }

  /**
   * Reset precision tracking to baseline state
   */
  private resetPrecisionTracking(): void {
    const now = Date.now();
    this.precisionTracking = {
      baselineStartTime: now,
      baselineCharIndex: this.currentCharIndex,
      seekOffset: 0,
      lastSyncPoint: now,
      charIndexHistory: [{ charIndex: this.currentCharIndex, timestamp: now }],
      realTimeWPM: this.options.wordsPerMinute || 200,
      averageCharPerSecond: 0,
    };
  }

  /**
   * Update precision tracking with current position
   */
  private updatePrecisionTracking(charIndex: number): void {
    const now = Date.now();

    // Add to history
    this.precisionTracking.charIndexHistory.push({
      charIndex,
      timestamp: now,
    });

    // Keep only last 10 entries for efficiency
    if (this.precisionTracking.charIndexHistory.length > 10) {
      this.precisionTracking.charIndexHistory.shift();
    }

    // Calculate real-time metrics
    this.calculateRealTimeMetrics();

    this.precisionTracking.lastSyncPoint = now;
  }

  /**
   * Calculate real-time speaking metrics for better estimation
   */
  private calculateRealTimeMetrics(): void {
    const history = this.precisionTracking.charIndexHistory;

    if (history.length < 2) return;

    const recent = history.slice(-3); // Use last 3 points
    if (recent.length < 2) return;

    const timeSpan = recent[recent.length - 1].timestamp - recent[0].timestamp;
    const charSpan = recent[recent.length - 1].charIndex - recent[0].charIndex;

    if (timeSpan > 0 && charSpan > 0) {
      this.precisionTracking.averageCharPerSecond =
        (charSpan / timeSpan) * 1000;

      // Estimate WPM based on character rate
      const avgCharsPerWord = this.currentText
        ? this.currentText.length / this.currentText.split(/\s+/).length
        : 5;

      this.precisionTracking.realTimeWPM =
        (this.precisionTracking.averageCharPerSecond * 60) / avgCharsPerWord;
    }
  }

  /**
   * Enhanced highlighting with synchronization fallbacks - Auto-scroll disabled
   * Uses word-based indexing for accurate synchronization
   */
  private highlightWithSync(processedCharIndex: number): void {
    try {
      // Get word index from processed char index
      const wordIndex = this.getWordIndexFromProcessedCharIndex(processedCharIndex);

      // Use word-based highlighting for accuracy
      this.highlighter.highlightWordByIndex(wordIndex);

      this.startBackupHighlighting();
    } catch {
      this.fallbackHighlighting(processedCharIndex);
    }
  }

  /**
   * Start backup highlighting for sync correction
   */
  private startBackupHighlighting(): void {
    this.stopBackupHighlighting();

    this.highlightSyncTimer = window.setInterval(() => {
      if (this.isPlaying && !this.isPaused) {
        // Check if word boundary events are falling behind
        const timeSinceLastUpdate = Date.now() - this.lastCharIndexUpdate;

        // If no word boundary event for 2 seconds, use estimated position
        if (timeSinceLastUpdate > 2000) {
          const estimatedIndex = this.estimateCurrentCharIndex();
          if (estimatedIndex > this.currentCharIndex) {
            // Use word-based highlighting for accuracy
            const wordIndex = this.getWordIndexFromProcessedCharIndex(estimatedIndex);
            this.highlighter.highlightWordByIndex(wordIndex);
            this.currentCharIndex = estimatedIndex;
            this.lastCharIndexUpdate = Date.now();
          }
        }
      }
    }, 500); // Check every 500ms
  }

  /**
   * Stop backup highlighting timer
   */
  private stopBackupHighlighting(): void {
    if (this.highlightSyncTimer) {
      clearInterval(this.highlightSyncTimer);
      this.highlightSyncTimer = null;
    }
  }

  /**
   * Estimate current character index based on time and WPM - Enhanced for seeking
   */
  private estimateCurrentCharIndexFromTime(): number {
    if (!this.isPlaying || this.isPaused || !this.currentText)
      return this.currentCharIndex;

    const elapsed = Date.now() - this.startTime - this.totalPausedDuration;
    const elapsedMinutes = elapsed / 1000 / 60;

    // Use actual WPM if available, otherwise use configured WPM
    const wpm =
      this.estimatedWPM > 0
        ? this.estimatedWPM
        : this.options.wordsPerMinute || 200;
    const wordsSpoken = wpm * elapsedMinutes;

    // Convert words to character index (approximate)
    const avgCharsPerWord =
      this.currentText.length / this.currentText.split(/\s+/).length;
    const estimatedCharIndex = Math.floor(wordsSpoken * avgCharsPerWord);

    return Math.min(estimatedCharIndex, this.currentText.length - 1);
  }

  /**
   * Estimate current character index based on time and WPM
   */
  private estimateCurrentCharIndex(): number {
    return this.estimateCurrentCharIndexFromTime();
  }

  /**
   * Get word count up to a specific character index
   */
  private getWordCountUpToIndex(charIndex: number): number {
    if (!this.currentText || charIndex <= 0) return 0;

    const textUpToIndex = this.currentText.substring(0, charIndex);
    return textUpToIndex.split(/\s+/).filter((word) => word.length > 0).length;
  }

  /**
   * Fallback highlighting method - No auto-scroll
   */
  private fallbackHighlighting(charIndex: number): void {
    try {
      // Simple word highlighting fallback without auto-scroll
      const wordAtIndex = this.getWordAtIndex(charIndex);
      if (wordAtIndex) {
        // Find the word element and highlight it manually
        const wordElements =
          this.container.querySelectorAll("[data-word-index]");
        wordElements.forEach((element) => {
          const elementIndex = parseInt(
            element.getAttribute("data-word-index") || "0"
          );
          if (Math.abs(elementIndex - charIndex) < 10) {
            // Close enough
            element.classList.add("highlighted-word");
            // Auto-scroll removed
          } else {
            element.classList.remove("highlighted-word");
          }
        });
      }
    } catch {
      // Fallback highlighting failed silently
    }
  }

  /**
   * Get current progress percentage - Enhanced for seek synchronization
   * Uses word-based calculation for accurate progress
   */
  public getCurrentProgress(): number {
    if (this.totalWordCount === 0) return 0;

    // Get current word index from processed char position
    const currentWordIndex = this.getWordIndexFromProcessedCharIndex(
      this.currentCharIndex
    );

    // Calculate progress based on word count (more accurate than character count)
    const wordProgress = (currentWordIndex / this.totalWordCount) * 100;

    // Ensure progress is within bounds
    const clampedProgress = Math.max(0, Math.min(100, wordProgress));

    return clampedProgress;
  }

  /**
   * Start progress tracking - Optimized for seeking accuracy
   */
  private startProgressTracking(): void {
    this.stopProgressTracking();

    this.progressTimer = window.setInterval(() => {
      if (this.isPlaying && !this.isPaused) {
        const progress = this.getCurrentProgress();
        this.events.onProgress?.(progress);
      }
    }, 100); // Balanced frequency for smooth updates without performance issues
  }

  /**
   * Stop progress tracking
   */
  private stopProgressTracking(): void {
    if (this.progressTimer) {
      clearInterval(this.progressTimer);
      this.progressTimer = null;
    }
  }

  /**
   * Estimate reading duration based on words per minute
   */
  private estimateDuration(): number {
    const words = this.currentText.split(/\s+/).length;
    const wordsPerMinute = this.options.wordsPerMinute || 200;
    return (words / wordsPerMinute) * 60 * 1000; // Convert to milliseconds
  }

  /**
   * Get word at specific character index
   */
  private getWordAtIndex(charIndex: number): string {
    const words = this.currentText.split(/\s+/);
    let currentIndex = 0;

    for (const word of words) {
      if (charIndex >= currentIndex && charIndex < currentIndex + word.length) {
        return word;
      }
      currentIndex += word.length + 1; // +1 for space
    }

    return "";
  }

  /**
   * Handle pausing after headings for better reading flow - Simplified approach
   */
  private handleHeadingPause(charIndex: number): void {
    // Simplified heading pause logic using text-based detection instead of DOM manipulation
    // This avoids timing issues with speechSynthesis.pause/resume

    if (this.isWaitingAfterHeading) return;

    // Look for heading patterns in the surrounding text
    const contextStart = Math.max(0, charIndex - 50);
    const contextEnd = Math.min(this.currentText.length, charIndex + 50);
    const contextText = this.currentText.substring(contextStart, contextEnd);

    // Check if we just passed a heading-like pattern
    const headingPattern = /(#+\s*[^\n\r]+|<h[1-6][^>]*>.*?<\/h[1-6]>)/i;
    const headingMatch = contextText.match(headingPattern);

    if (headingMatch) {
      const headingEnd =
        contextStart + (headingMatch.index || 0) + headingMatch[0].length;

      // If we're just past a heading and haven't processed this one yet
      if (
        charIndex >= headingEnd &&
        charIndex <= headingEnd + 10 &&
        this.lastHeadingEndIndex !== headingEnd
      ) {
        this.lastHeadingEndIndex = headingEnd;
        this.isWaitingAfterHeading = true;

        // Mark that we processed this heading and reset flag after a delay
        setTimeout(() => {
          this.isWaitingAfterHeading = false;
        }, 500); // Shorter delay since we're using text-based pauses now
      }
    }
  }

  /**
   * Get current playing state
   */
  public getState(): {
    isPlaying: boolean;
    isPaused: boolean;
    progress: number;
    currentText: string;
    duration: number;
    remainingTime: number;
  } {
    const progress = this.getCurrentProgress();
    const duration = this.getTotalDuration();
    const remainingTime = this.getRemainingTime();

    return {
      isPlaying: this.isPlaying,
      isPaused: this.isPaused,
      progress,
      currentText: this.currentText,
      duration,
      remainingTime,
    };
  }

  /**
   * Update options
   */
  public updateOptions(newOptions: Partial<SpeechReaderOptions>): void {
    this.options = { ...this.options, ...newOptions };

    if (this.utterance) {
      this.configureSpeech();
    }

    // Update highlighter colors if changed
    if (newOptions.highlightColors) {
      this.highlighter = new TextHighlighter(
        this.container,
        newOptions.highlightColors
      );
    }
  }

  /**
   * Get available voices
   */
  public getAvailableVoices(): SpeechSynthesisVoice[] {
    return speechSynthesis.getVoices();
  }

  /**
   * Set voice
   */
  public setVoice(voice: SpeechSynthesisVoice): void {
    this.options.voice = voice;
    if (this.utterance) {
      this.utterance.voice = voice;
    }
  }

  /**
   * Clean up resources
   */
  public destroy(): void {
    this.stop();
    this.stopProgressTracking();
    this.stopBackupHighlighting();
    this.highlighter.destroy();
  }

  /**
   * Enhanced seek validation to ensure accuracy
   */
  private validateSeekPosition(charIndex: number): boolean {
    // Validate that the character index is within bounds
    if (charIndex < 0 || charIndex >= this.currentText.length) {
      return false;
    }

    const char = this.currentText[charIndex];
    if (!char || char === "\0") {
      return false;
    }

    return true;
  }
}
