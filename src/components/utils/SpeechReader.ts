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
  private wordMappings: Array<{
    wordIndex: number;
    charStart: number;
    charEnd: number;
    wordText: string;
  }> = [];

  // Seeking state management
  private seekingState = {
    isInSeekMode: false,
    targetPercentage: 0,
    targetCharIndex: 0,
    seekStartTime: 0,
    preSeekCharIndex: 0,
  };

  constructor(
    container: HTMLElement,
    options: SpeechReaderOptions,
    events: SpeechReaderEvents = {}
  ) {
    console.log("SpeechReader constructor called");

    // Check for Web Speech API support
    if (!("speechSynthesis" in window)) {
      console.error("Web Speech API is not supported in this browser");
      throw new Error("Web Speech API is not supported in this browser");
    }

    console.log("Web Speech API is supported");

    this.container = container;
    this.options = {
      wordsPerMinute: 200, // Increased for better speed while maintaining clarity
      autoScroll: false, // Disabled auto-scroll to prevent screen jumping
      pitch: 1.0, // Natural pitch
      rate: 0.85, // Optimized rate for speed and clarity balance
      volume: 1,
      ...options,
    };
    this.events = events;

    console.log("Creating TextHighlighter with container:", container);
    console.log("Highlight colors:", options.highlightColors);

    try {
      this.highlighter = new TextHighlighter(
        container,
        options.highlightColors
      );
      console.log("TextHighlighter created successfully");
    } catch (error) {
      console.error("Error creating TextHighlighter:", error);
      throw error;
    }

    this.setupSpeechSynthesis();
  }

  /**
   * Split text into segments for seeking capability
   */
  private splitTextIntoSegments(): void {
    // Clear any existing segments first
    this.textSegments = [];

    if (!this.currentText || this.currentText.trim() === "") {
      console.warn("No text to split into segments");
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
          } else {
            console.warn("Duplicate segment detected and skipped:", segmentKey);
          }
        } else {
          console.warn(
            "Could not find sentence position:",
            sentence.substring(0, 50)
          );
        }
      }
    }

    console.log(
      `Created ${this.textSegments.length} text segments from ${this.currentText.length} characters`
    );
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

      // Log voice selection for debugging
      console.log(
        "Configured voice:",
        voice?.name,
        "Rate:",
        this.utterance.rate
      );
    }
  }
  /**
   * Get preferred voice (prioritize English voices)
   */
  private getPreferredVoice(): SpeechSynthesisVoice | null {
    const voices = speechSynthesis.getVoices();

    // Try to find high-quality natural English voices first
    const preferredVoices = [
      "Google US English", // Google voices are usually high quality
      "Microsoft Aria Online (Natural) - English (United States)", // Microsoft natural voices
      "Microsoft Guy Online (Natural) - English (United States)",
      "Alex", // MacOS default, usually good quality
      "Samantha", // MacOS alternative
      "Google UK English Female",
      "Google UK English Male",
      "Microsoft David - English (United States)", // Microsoft voices
      "Microsoft Zira - English (United States)",
    ];

    for (const voiceName of preferredVoices) {
      const voice = voices.find(
        (v) => v.name.includes(voiceName) || v.name === voiceName
      );
      if (voice && voice.lang.startsWith("en")) {
        console.log("Selected preferred voice:", voice.name);
        return voice;
      }
    }

    // Fallback to any high-quality English voice
    const englishVoices = voices.filter((v) => v.lang.startsWith("en"));

    // Prefer online/cloud voices as they're usually higher quality
    const onlineVoice = englishVoices.find(
      (v) =>
        v.name.toLowerCase().includes("online") ||
        v.name.toLowerCase().includes("neural") ||
        v.name.toLowerCase().includes("enhanced")
    );

    if (onlineVoice) {
      console.log("Selected online voice:", onlineVoice.name);
      return onlineVoice;
    }

    // Fallback to first English voice
    const fallbackVoice = englishVoices[0] || voices[0] || null;
    if (fallbackVoice) {
      console.log("Selected fallback voice:", fallbackVoice.name);
    }

    return fallbackVoice;
  }

  /**
   * Process text to add natural pauses for punctuation
   */
  private processTextForNaturalSpeech(text: string): string {
    if (!text) return "";

    // Enhanced text processing for much smoother speech with natural pauses
    let processedText = text;

    // Remove excessive whitespace first
    processedText = processedText.replace(/\s+/g, " ");

    // Add longer pauses after periods, exclamation marks, and question marks
    processedText = processedText.replace(
      /([.!?]+)\s*/g,
      (match, punctuation) => {
        // Add substantial pause for sentence endings - use SSML-like approach
        if (
          punctuation.includes(".") ||
          punctuation.includes("!") ||
          punctuation.includes("?")
        ) {
          return punctuation + "     "; // Multiple spaces for longer pause
        }
        return match;
      }
    );

    // Add medium pauses after colons and semicolons
    processedText = processedText.replace(/([;:]+)\s*/g, "$1   ");

    // Add short pauses after commas and dashes
    processedText = processedText.replace(/([,—–-]+)\s*/g, "$1  ");

    // Add pauses after numbered lists and bullet points
    processedText = processedText.replace(/(\d+\.|\*|\-|\•)\s*/g, "$1  ");

    // Handle parenthetical expressions with slight pauses
    processedText = processedText.replace(/\(\s*([^)]+)\s*\)/g, "  ($1)  ");

    // Add pause before and after quotations
    processedText = processedText.replace(
      /["'"']([^"'"']*?)["'"']/g,
      '  "$1"  '
    );

    // Handle section breaks and paragraph transitions with longer pauses
    processedText = processedText.replace(/\n\s*\n/g, "     "); // Paragraph breaks

    // Add pauses after headers (markdown style and HTML style) - Enhanced for better timing
    processedText = processedText.replace(/(#+\s*[^\n\r]+)/g, "$1        "); // Markdown headers with longer pause
    processedText = processedText.replace(
      /(<h[1-6][^>]*>.*?<\/h[1-6]>)/gi,
      "$1        "
    ); // HTML headers with longer pause

    // Add pauses after common heading patterns (words ending with : followed by content)
    processedText = processedText.replace(/([A-Z][^.!?:]*:)\s*/g, "$1      ");

    // Clean up excessive spaces but preserve intentional pause spaces
    processedText = processedText.replace(/ {9,}/g, "        "); // Max 8 spaces for extra long pauses (after headings)
    processedText = processedText.replace(/ {6,7}/g, "      "); // 6 spaces for long pauses
    processedText = processedText.replace(/ {4,5}/g, "    "); // Normalize to 4 spaces for medium pauses
    processedText = processedText.replace(/ {2,3}/g, "  "); // Normalize to 2 spaces for short pauses

    console.log("Original text length:", text.length);
    console.log("Processed text length:", processedText.length);
    console.log("Sample processed text:", processedText.substring(0, 200));

    return processedText.trim();
  }

  /**
   * Start reading the text
   */
  public start(): void {
    console.log("SpeechReader.start() called");

    if (this.isPlaying) {
      console.log("Already playing, stopping first");
      this.stop();
    }

    const rawText = this.highlighter.getReadableText();
    console.log("Raw readable text length:", rawText?.length);

    if (!rawText) {
      console.error("No readable text found");
      this.events.onError?.(new Error("No readable text found"));
      return;
    }

    // Process text for natural speech with punctuation pauses
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

    // Set the new text and split into segments
    this.currentText = text;
    this.splitTextIntoSegments();

    console.log("Text segments created:", this.textSegments.length);
    console.log(
      "First segment preview:",
      this.textSegments[0]?.text?.substring(0, 100)
    );

    this.utterance = new SpeechSynthesisUtterance(text);
    this.configureSpeech();

    console.log("Speech synthesis available:", "speechSynthesis" in window);
    console.log("Setting up event handlers...");

    // Setup event handlers
    this.utterance.onstart = () => {
      console.log("Speech started");
      this.isPlaying = true;
      this.isPaused = false;
      this.startTime = Date.now();
      this.totalPausedDuration = 0;
      this.startProgressTracking();
      this.events.onStart?.();
    };

    this.utterance.onend = () => {
      console.log("Speech ended");
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

      // Log only unexpected errors
      console.error("Speech synthesis error:", error);
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

    // Start speaking
    console.log("Starting speech synthesis...");
    console.log("speechSynthesis.speaking:", speechSynthesis.speaking);
    console.log("speechSynthesis.pending:", speechSynthesis.pending);
    console.log("speechSynthesis.paused:", speechSynthesis.paused);

    try {
      // Cancel any ongoing speech first
      if (speechSynthesis.speaking || speechSynthesis.pending) {
        console.log("Canceling ongoing speech...");
        speechSynthesis.cancel();
        // Wait a bit longer for the cancellation to complete
        setTimeout(() => {
          speechSynthesis.speak(this.utterance!);
          console.log(
            "speechSynthesis.speak() called successfully after cancellation"
          );
        }, 150); // Increased delay for better cancellation handling
      } else {
        speechSynthesis.speak(this.utterance);
        console.log("speechSynthesis.speak() called successfully");
      }
    } catch (error) {
      console.error("Error calling speechSynthesis.speak():", error);
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

    console.log("Speech stopped and state reset");
  }

  /**
   * Seek to a specific position (0-100%) - Optimized for accurate sync
   */
  public seekTo(percentage: number): void {
    if (!this.currentText || this.textSegments.length === 0) {
      console.warn("No text loaded for seeking");
      return;
    }

    // Clamp percentage between 0 and 100
    percentage = Math.max(0, Math.min(100, percentage));

    console.log(`[SpeechReader] Seeking to ${percentage}%`);

    // Enhanced character index calculation with segment-aware targeting
    let targetCharIndex = this.calculatePreciseTargetIndex(percentage);
    targetCharIndex = this.findNearestWordBoundary(targetCharIndex);

    // Find the segment that contains this character index
    const targetSegmentIndex = this.findSegmentByCharIndex(targetCharIndex);

    if (targetSegmentIndex === -1) {
      console.warn("Could not find segment for target position");
      return;
    }

    // Store current state for restoration
    const wasPlaying = this.isPlaying;
    const wasPaused = this.isPaused;
    const preservedText = this.currentText;
    const preservedSegments = [...this.textSegments];
    const preservedWordMappings = [...this.wordMappings];

    // Cancel current speech and reset state
    this.cancelCurrentSpeech();

    // Restore preserved data
    this.currentText = preservedText;
    this.textSegments = preservedSegments;
    this.wordMappings = preservedWordMappings;

    // Update position state FIRST - Critical for sync
    this.updateSeekPosition(targetCharIndex, targetSegmentIndex, percentage);

    // Immediately update highlighting to match seek position
    this.synchronizeHighlightingToPosition(targetCharIndex);

    // Force immediate progress update to prevent UI lag
    this.forceProgressUpdate(percentage);

    // Create new utterance from target position if we have remaining text
    const remainingText = this.getRemainingTextFromIndex(targetCharIndex);
    if (remainingText && remainingText.trim()) {
      this.createSeekUtterance(remainingText, targetCharIndex);

      // Resume playback if it was active before seeking
      if (wasPlaying && !wasPaused) {
        this.resumePlaybackAfterSeek();
      }
    }

    // Trigger seek event with actual position achieved
    const actualPercentage = this.getCurrentProgress();
    this.events.onSeek?.(actualPercentage);

    console.log(
      `[SpeechReader] Seek completed: ${percentage}% -> ${actualPercentage}% (char: ${targetCharIndex})`
    );
  }

  /**
   * Calculate precise target index using segment-aware positioning
   */
  private calculatePreciseTargetIndex(percentage: number): number {
    if (this.textSegments.length === 0) {
      return Math.floor((percentage / 100) * this.currentText.length);
    }

    // Use segments for more accurate positioning
    const targetSegmentIndex = Math.floor(
      (percentage / 100) * this.textSegments.length
    );

    if (targetSegmentIndex >= this.textSegments.length) {
      return this.currentText.length - 1;
    }

    const targetSegment = this.textSegments[targetSegmentIndex];

    // Calculate position within the target segment
    const segmentProgress =
      (percentage / 100) * this.textSegments.length - targetSegmentIndex;
    const charWithinSegment = Math.floor(
      segmentProgress * (targetSegment.endIndex - targetSegment.startIndex)
    );

    return targetSegment.startIndex + charWithinSegment;
  }

  /**
   * Cancel current speech and clean up state
   */
  private cancelCurrentSpeech(): void {
    speechSynthesis.cancel();

    this.isPlaying = false;
    this.isPaused = false;
    this.stopProgressTracking();
    this.stopBackupHighlighting();

    // Clear utterance events to prevent conflicts
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
   */
  private synchronizeHighlightingToPosition(charIndex: number): void {
    try {
      // Clear any existing highlights first
      this.highlighter.clearAllHighlights();

      // Apply new highlighting immediately
      this.highlighter.highlightWordAtIndex(charIndex);

      console.log(
        `[SpeechReader] Highlighting synchronized to char ${charIndex}`
      );
    } catch (error) {
      console.warn("Failed to synchronize highlighting:", error);
      this.fallbackHighlighting(charIndex);
    }
  }

  /**
   * Force immediate progress update to prevent UI lag
   */
  private forceProgressUpdate(targetPercentage: number): void {
    // Immediately update progress to target percentage
    this.events.onProgress?.(targetPercentage);

    // Schedule follow-up updates to ensure accuracy
    setTimeout(() => {
      const actualProgress = this.getCurrentProgress();
      this.events.onProgress?.(actualProgress);
      console.log(
        `[SpeechReader] Progress sync: target=${targetPercentage}%, actual=${actualProgress}%`
      );
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
      console.warn("No utterance available for resuming playback");
      return;
    }

    try {
      speechSynthesis.speak(this.utterance);
      this.isPlaying = true;
      this.isPaused = false;
      this.startProgressTracking();
      this.startBackupHighlighting();

      console.log("[SpeechReader] Playback resumed after seek");
    } catch (error) {
      console.error("Error resuming speech after seek:", error);
      this.events.onError?.(new Error(`Failed to resume speech: ${error}`));
    }
  }

  /**
   * Recalibrate timing specifically for seeking operations
   */
  private recalibrateTimingForSeek(targetPercentage: number): void {
    const totalDuration = this.estimateDuration();
    const targetElapsedTime = (targetPercentage / 100) * totalDuration;

    // Set start time as if we've been playing from the beginning up to this point
    this.startTime = Date.now() - targetElapsedTime;
    this.totalPausedDuration = 0;
    this.estimatedWPM = 0; // Reset for recalculation

    console.log(
      `[SpeechReader] Timing recalibrated: ${targetPercentage}% (${Math.round(
        targetElapsedTime / 1000
      )}s elapsed)`
    );
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
   */
  private setupOptimizedSeekEventHandlers(startCharIndex: number): void {
    if (!this.utterance) return;

    this.utterance.onstart = () => {
      this.isPlaying = true;
      this.isPaused = false;
      this.startProgressTracking();
      this.startBackupHighlighting();
      this.events.onStart?.();
      console.log("[SpeechReader] Speech started after seek");
    };

    this.utterance.onend = () => {
      this.isPlaying = false;
      this.isPaused = false;
      this.highlighter.clearAllHighlights();
      this.stopProgressTracking();
      this.events.onEnd?.();
      console.log("[SpeechReader] Speech ended");
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

      console.error("Speech synthesis error after seek:", error);
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
        // Calculate precise character index relative to the original text
        const relativeCharIndex = event.charIndex;
        const actualCharIndex = startCharIndex + relativeCharIndex;

        if (this.validateSeekPosition(actualCharIndex)) {
          // Update position with enhanced synchronization
          this.updateCurrentPositionOptimized(actualCharIndex);

          // Apply highlighting immediately
          this.highlighter.highlightWordAtIndex(actualCharIndex);

          // Handle heading pauses
          this.handleHeadingPause(actualCharIndex);

          this.events.onWordHighlight?.(
            this.getWordAtIndex(actualCharIndex),
            actualCharIndex
          );
        } else {
          console.warn(
            `Invalid position after seek: ${actualCharIndex}, using fallback`
          );
          const estimatedIndex = Math.max(
            startCharIndex,
            this.estimateCurrentCharIndexFromTime()
          );
          if (this.validateSeekPosition(estimatedIndex)) {
            this.updateCurrentPositionOptimized(estimatedIndex);
            this.highlighter.highlightWordAtIndex(estimatedIndex);
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
  private findNearestWordBoundary(charIndex: number): number {
    if (!this.currentText || charIndex <= 0) return 0;
    if (charIndex >= this.currentText.length)
      return this.currentText.length - 1;

    // Look for word boundaries (spaces, punctuation) around the target index
    const start = Math.max(0, charIndex - 10);

    // Find the nearest space or word boundary before the target
    for (let i = charIndex; i >= start; i--) {
      const char = this.currentText[i];
      if (
        char === " " ||
        char === "\n" ||
        char === "\t" ||
        /[.!?]/.test(char)
      ) {
        return i + 1; // Start of next word
      }
    }

    return charIndex;
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
   * Setup event handlers for seek operation - FIXED version
   */
  private setupSeekEventHandlersFixed(startCharIndex: number): void {
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

      console.error("Speech synthesis error:", error);
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
        // CRITICAL FIX: Calculate accurate character index relative to original text
        const relativeCharIndex = event.charIndex;
        const actualCharIndex = startCharIndex + relativeCharIndex;

        console.log(
          `Word boundary: relative=${relativeCharIndex}, actual=${actualCharIndex}, start=${startCharIndex}`
        );

        // Validate the calculated index to prevent errors
        if (this.validateSeekPosition(actualCharIndex)) {
          // Update current position with proper timing info
          this.updateCurrentPosition(actualCharIndex);

          // Handle heading pause logic
          this.handleHeadingPause(actualCharIndex);

          // Enhanced highlighting with better synchronization
          this.highlightWithSync(actualCharIndex);

          this.events.onWordHighlight?.(
            this.getWordAtIndex(actualCharIndex),
            actualCharIndex
          );
        } else {
          // Enhanced fallback with better estimation
          console.warn(
            `Invalid seek position ${actualCharIndex}, using enhanced fallback`
          );
          const estimatedIndex = Math.max(
            startCharIndex,
            this.estimateCurrentCharIndexFromTime()
          );
          if (this.validateSeekPosition(estimatedIndex)) {
            this.updateCurrentPosition(estimatedIndex);
            this.highlightWithSync(estimatedIndex);
          }
        }
      }
    };
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
   * Enhanced speech smoothing for better flow and natural pauses
   */
  private enhanceSpeechSmoothing(): void {
    if (!this.utterance) return;

    // Apply additional smoothing settings for better flow
    this.utterance.rate = Math.max(
      0.6,
      Math.min(1.0, this.options.rate || 0.75)
    );
    this.utterance.pitch = Math.max(
      0.8,
      Math.min(1.2, this.options.pitch || 1.0)
    );

    // Ensure volume is optimal for clarity
    this.utterance.volume = Math.max(
      0.8,
      Math.min(1.0, this.options.volume || 1.0)
    );

    console.log("Enhanced speech smoothing applied:", {
      rate: this.utterance.rate,
      pitch: this.utterance.pitch,
      volume: this.utterance.volume,
    });
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
   */
  private buildWordMappings(): void {
    this.wordMappings = [];

    if (!this.currentText) return;

    const words = this.currentText.split(/(\s+)/); // Include spaces in split
    let charPosition = 0;
    let wordIndex = 0;

    for (const word of words) {
      if (word.trim()) {
        // Only process actual words, skip whitespace
        this.wordMappings.push({
          wordIndex,
          charStart: charPosition,
          charEnd: charPosition + word.length - 1,
          wordText: word,
        });
        wordIndex++;
      }
      charPosition += word.length;
    }

    console.log(
      `Built ${this.wordMappings.length} word mappings for ${this.currentText.length} characters`
    );
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
   */
  private highlightWithSync(charIndex: number): void {
    try {
      // Primary highlighting method only, no auto-scroll
      this.highlighter.highlightWordAtIndex(charIndex);

      // Start/update backup highlighting timer for sync correction
      this.startBackupHighlighting();
    } catch (error) {
      console.warn("Primary highlighting failed, using fallback:", error);
      // Fallback highlighting method
      this.fallbackHighlighting(charIndex);
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
            this.highlighter.highlightWordAtIndex(estimatedIndex);
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
    } catch (error) {
      console.warn("Fallback highlighting also failed:", error);
    }
  }

  /**
   * Get current progress percentage - Enhanced for seek synchronization
   */
  public getCurrentProgress(): number {
    if (!this.currentText || this.currentText.length === 0) return 0;

    // For more accurate progress, always use character-based calculation
    const characterProgress =
      (this.currentCharIndex / this.currentText.length) * 100;

    // Ensure progress is within bounds and smooth
    const clampedProgress = Math.max(0, Math.min(100, characterProgress));

    // Add small smoothing for better UI experience
    if (this.isPlaying && !this.isPaused) {
      // Use real-time estimation if available for smoothing
      const estimatedIndex = this.estimateCurrentCharIndexFromTime();
      if (
        Math.abs(estimatedIndex - this.currentCharIndex) <
        this.currentText.length * 0.02
      ) {
        // Only use estimation if within 2% of current position (prevents jumps)
        const estimatedProgress =
          (estimatedIndex / this.currentText.length) * 100;
        return Math.max(clampedProgress, Math.min(100, estimatedProgress));
      }
    }

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
   * Highlight text by character index - No auto-scroll
   */
  private highlightByCharIndex(charIndex: number): void {
    this.highlighter.highlightWordAtIndex(charIndex);
    // Auto-scroll removed for better user experience
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
      console.warn(`Invalid character index for seeking: ${charIndex}`);
      return false;
    }

    // Enhanced validation - check if we're at a reasonable text position
    const char = this.currentText[charIndex];
    if (!char || char === "\0") {
      console.warn(`Invalid character at index: ${charIndex}`);
      return false;
    }

    return true;
  }
}
