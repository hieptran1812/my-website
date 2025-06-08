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
      wordsPerMinute: 200,
      autoScroll: true,
      pitch: 1,
      rate: 1,
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
    this.textSegments = [];
    const sentences = this.currentText.split(/[.!?]+\s*/);
    let currentIndex = 0;

    for (const sentence of sentences) {
      if (sentence.trim()) {
        const trimmedSentence = sentence.trim();
        const startIndex = this.currentText.indexOf(
          trimmedSentence,
          currentIndex
        );
        const endIndex = startIndex + trimmedSentence.length;

        this.textSegments.push({
          text: trimmedSentence,
          startIndex,
          endIndex,
        });

        currentIndex = endIndex;
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
      this.utterance.voice = this.options.voice || this.getPreferredVoice();
      this.utterance.pitch = this.options.pitch || 1;
      this.utterance.rate = this.options.rate || 1;
      this.utterance.volume = this.options.volume || 1;
    }
  }

  /**
   * Get preferred voice (prioritize English voices)
   */
  private getPreferredVoice(): SpeechSynthesisVoice | null {
    const voices = speechSynthesis.getVoices();

    // Try to find a high-quality English voice
    const preferredVoices = [
      "Google US English",
      "Microsoft Edge English",
      "Alex",
      "Samantha",
      "Google UK English Female",
      "Google UK English Male",
    ];

    for (const voiceName of preferredVoices) {
      const voice = voices.find((v) => v.name.includes(voiceName));
      if (voice) return voice;
    }

    // Fallback to any English voice
    const englishVoice = voices.find((v) => v.lang.startsWith("en"));
    return englishVoice || voices[0] || null;
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

    const text = this.highlighter.getReadableText();
    console.log("Readable text length:", text?.length);

    if (!text) {
      console.error("No readable text found");
      this.events.onError?.(new Error("No readable text found"));
      return;
    }

    this.currentText = text;
    this.splitTextIntoSegments();
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
        // Wait a bit for the cancellation to complete
        setTimeout(() => {
          speechSynthesis.speak(this.utterance!);
          console.log(
            "speechSynthesis.speak() called successfully after cancellation"
          );
        }, 100);
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
   * Seek to a specific position (0-100%)
   */
  public seekTo(percentage: number): void {
    if (!this.currentText || this.textSegments.length === 0) {
      console.warn("No text loaded for seeking");
      return;
    }

    // Clamp percentage between 0 and 100
    percentage = Math.max(0, Math.min(100, percentage));

    // Calculate target character index based on percentage
    let targetCharIndex = Math.floor(
      (percentage / 100) * this.currentText.length
    );

    // Find the nearest word boundary for more accurate seeking
    targetCharIndex = this.findNearestWordBoundary(targetCharIndex);

    // Find the segment that contains this character index
    const targetSegmentIndex = this.findSegmentByCharIndex(targetCharIndex);

    if (targetSegmentIndex === -1) {
      console.warn("Could not find segment for target position");
      return;
    }

    const wasPlaying = this.isPlaying;
    const wasPaused = this.isPaused;

    // Stop current speech and clear timers
    this.stop();

    // Update current position with enhanced tracking - CRITICAL for sync
    this.currentCharIndex = targetCharIndex;
    this.currentSegmentIndex = targetSegmentIndex;
    this.lastCharIndexUpdate = Date.now();

    // Recalculate timing to match the seek position
    const totalDuration = this.estimateDuration();
    const elapsedDuration = (percentage / 100) * totalDuration;
    this.startTime = Date.now() - elapsedDuration;
    this.totalPausedDuration = 0;

    // Enhanced highlighting at the target position
    this.highlightWithSync(targetCharIndex);

    // Create new utterance from the target position
    const remainingText = this.getRemainingTextFromIndex(targetCharIndex);
    if (remainingText) {
      this.utterance = new SpeechSynthesisUtterance(remainingText);
      this.configureSpeech();
      this.setupSeekEventHandlers(targetCharIndex);

      // Recalibrate timing for accurate progress tracking
      this.recalibrateAfterSeek(percentage);

      // Resume playback if it was playing before
      if (wasPlaying && !wasPaused) {
        try {
          speechSynthesis.speak(this.utterance);
          this.isPlaying = true;
          this.isPaused = false;
          this.startProgressTracking();
          this.startBackupHighlighting();
        } catch (error) {
          console.error("Error resuming speech after seek:", error);
          this.events.onError?.(new Error(`Failed to resume speech: ${error}`));
        }
      } else {
        // If not playing, just update the visual state
        this.isPlaying = false;
        this.isPaused = false;
      }
    }

    // Trigger seek event
    this.events.onSeek?.(percentage);
  }

  /**
   * Find segment index by character index
   */
  private findSegmentByCharIndex(charIndex: number): number {
    for (let i = 0; i < this.textSegments.length; i++) {
      const segment = this.textSegments[i];
      if (charIndex >= segment.startIndex && charIndex <= segment.endIndex) {
        return i;
      }
    }
    return -1;
  }

  /**
   * Find the nearest word boundary for more accurate seeking
   */
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
   * Setup event handlers for seek operation
   */
  private setupSeekEventHandlers(startCharIndex: number): void {
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
        // Calculate actual character index relative to original text
        const actualCharIndex = startCharIndex + event.charIndex;

        // Update current position with timing info
        this.updateCurrentPosition(actualCharIndex);

        // Handle heading pause logic
        this.handleHeadingPause(actualCharIndex);

        // Enhanced highlighting with better synchronization
        this.highlightWithSync(actualCharIndex);

        this.events.onWordHighlight?.(
          this.getWordAtIndex(actualCharIndex),
          actualCharIndex
        );
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
  }

  /**
   * Enhanced highlighting with synchronization fallbacks
   */
  private highlightWithSync(charIndex: number): void {
    try {
      // Primary highlighting method
      this.highlighter.highlightWordAtIndex(charIndex);

      // Auto-scroll if enabled
      if (this.options.autoScroll) {
        this.highlighter.scrollToCurrentWord();
      }

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
   * Estimate current character index based on time and WPM
   */
  private estimateCurrentCharIndex(): number {
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
   * Get word count up to a specific character index
   */
  private getWordCountUpToIndex(charIndex: number): number {
    if (!this.currentText || charIndex <= 0) return 0;

    const textUpToIndex = this.currentText.substring(0, charIndex);
    return textUpToIndex.split(/\s+/).filter((word) => word.length > 0).length;
  }

  /**
   * Fallback highlighting method
   */
  private fallbackHighlighting(charIndex: number): void {
    try {
      // Simple word highlighting fallback
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
            if (this.options.autoScroll) {
              element.scrollIntoView({ behavior: "smooth", block: "center" });
            }
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
   * Get current progress percentage - optimized for better synchronization
   */
  public getCurrentProgress(): number {
    if (!this.isPlaying || this.isPaused) return 0;

    // Use character-based progress for better accuracy
    if (this.currentText.length > 0) {
      const characterProgress =
        (this.currentCharIndex / this.currentText.length) * 100;
      return Math.min(characterProgress, 100);
    }

    // Fallback to time-based calculation
    const elapsed = Date.now() - this.startTime - this.totalPausedDuration;
    const estimatedDuration = this.estimateDuration();
    return Math.min((elapsed / estimatedDuration) * 100, 100);
  }

  /**
   * Start progress tracking
   */
  private startProgressTracking(): void {
    this.stopProgressTracking();

    this.progressTimer = window.setInterval(() => {
      if (this.isPlaying && !this.isPaused) {
        const progress = this.getCurrentProgress();
        this.events.onProgress?.(progress);
      }
    }, 100);
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
   * Highlight text by character index
   */
  private highlightByCharIndex(charIndex: number): void {
    this.highlighter.highlightWordAtIndex(charIndex);

    if (this.options.autoScroll) {
      this.highlighter.scrollToCurrentWord();
    }
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
   * Handle pausing after headings for better reading flow
   */
  private handleHeadingPause(charIndex: number): void {
    if (this.isWaitingAfterHeading) return;

    // Get the current word and paragraph information
    const wordInfo = this.highlighter.getWordByCharacterIndex(charIndex);
    if (!wordInfo) return;

    const paragraphInfo =
      this.highlighter.getParagraphByCharacterIndex(charIndex);
    if (!paragraphInfo) return;

    const tagName = paragraphInfo.element.tagName.toLowerCase();

    // Check if we're in a heading
    if (tagName.match(/^h[1-6]$/)) {
      // Check if this is the last word in the heading
      const isLastWordInHeading =
        wordInfo.endIndex >= paragraphInfo.endIndex - 1;

      if (
        isLastWordInHeading &&
        this.lastHeadingEndIndex !== paragraphInfo.endIndex
      ) {
        this.lastHeadingEndIndex = paragraphInfo.endIndex;
        this.isWaitingAfterHeading = true;

        // Pause the speech
        speechSynthesis.pause();

        // Resume after 1.5s
        setTimeout(() => {
          if (this.isPlaying && this.isWaitingAfterHeading) {
            speechSynthesis.resume();
            this.isWaitingAfterHeading = false;
          }
        }, 1500);
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
   * Recalibrate timing and progress after seeking operation
   */
  private recalibrateAfterSeek(targetPercentage: number): void {
    // Calculate the accurate time adjustment for seeking
    const totalDuration = this.estimateDuration();
    const targetElapsedTime = (targetPercentage / 100) * totalDuration;

    // Set the start time as if we had been playing from the beginning
    this.startTime = Date.now() - targetElapsedTime;
    this.totalPausedDuration = 0;
    this.lastCharIndexUpdate = Date.now();

    // Reset estimated WPM to recalculate from this point
    this.estimatedWPM = 0;

    // Force progress update immediately
    if (this.isPlaying && !this.isPaused) {
      const progress = this.getCurrentProgress();
      this.events.onProgress?.(progress);
    }
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

    // Validate that we can find a word at this position
    const wordAtIndex = this.getWordAtIndex(charIndex);
    if (!wordAtIndex) {
      console.warn(`No word found at character index: ${charIndex}`);
      return false;
    }

    return true;
  }
}
