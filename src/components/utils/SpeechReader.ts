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
  private lastHeadingEndIndex = -1;
  private isWaitingAfterHeading = false;

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
      this.startTime += Date.now() - this.pausedTime;
      this.startProgressTracking();
      this.events.onResume?.();
    };

    // Handle word boundary events for highlighting
    this.utterance.onboundary = (event) => {
      if (event.name === "word") {
        // Handle heading pause logic
        this.handleHeadingPause(event.charIndex);

        this.highlighter.highlightWordAtIndex(event.charIndex);
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
    }
  }

  /**
   * Resume the speech
   */
  public resume(): void {
    if (this.isPlaying && this.isPaused) {
      speechSynthesis.resume();
    }
  }

  /**
   * Stop the speech
   */
  public stop(): void {
    speechSynthesis.cancel();
    this.isPlaying = false;
    this.isPaused = false;
    this.highlighter.clearAllHighlights();
    this.stopProgressTracking();
  }

  /**
   * Start progress tracking
   */
  private startProgressTracking(): void {
    this.stopProgressTracking();

    this.progressTimer = window.setInterval(() => {
      if (this.isPlaying && !this.isPaused) {
        const elapsed = Date.now() - this.startTime;
        const estimatedDuration = this.estimateDuration();
        const progress = Math.min((elapsed / estimatedDuration) * 100, 100);
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
  } {
    const elapsed = this.isPlaying ? Date.now() - this.startTime : 0;
    const estimatedDuration = this.estimateDuration();
    const progress =
      estimatedDuration > 0
        ? Math.min((elapsed / estimatedDuration) * 100, 100)
        : 0;

    return {
      isPlaying: this.isPlaying,
      isPaused: this.isPaused,
      progress,
      currentText: this.currentText,
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
    this.highlighter.destroy();
  }
}
