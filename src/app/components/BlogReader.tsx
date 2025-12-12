"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { useTheme } from "../ThemeProvider";
import CollectionTag from "../../components/CollectionTag";
import "katex/dist/katex.min.css";
import MathJax from "./MathJax";
import "./BlogContent.css"; // Re-enabled for proper styling
import "../../components/styles/TextHighlight.css"; // Import text highlighting styles
import { SpeechReader } from "../../components/utils/SpeechReader";
import type {
  SpeechReaderOptions,
  SpeechReaderEvents,
} from "../../components/utils/SpeechReader";
import BlogShareSection from "./BlogShareSection";
import { formatDate } from "../../lib/dateUtils";

interface TocItem {
  id: string;
  title: string;
  titleHtml: string;
  level: number;
  element: HTMLElement;
}

interface BlogReaderProps {
  children?: React.ReactNode;
  title: string;
  publishDate: string;
  readTime?: string;
  tags?: string[];
  category?: string;
  author?: string;
  postSlug?: string;
  collection?: string;
  dangerouslySetInnerHTML?: { __html: string };
}

export default function BlogReader({
  children,
  title,
  publishDate,
  readTime = "5 min read",
  tags = [],
  category = "Article",
  author = "Hiep Tran",
  postSlug,
  collection,
  dangerouslySetInnerHTML,
}: BlogReaderProps) {
  const { theme, isReadingMode, setReadingMode } = useTheme();
  const [fontSize, setFontSize] = useState(18);
  const [lineHeight, setLineHeight] = useState(1.6);
  const [tocItems, setTocItems] = useState<TocItem[]>([]);
  const [activeSection, setActiveSection] = useState<string>("");
  const [showToc, setShowToc] = useState(true);
  const [tocCollapsed, setTocCollapsed] = useState(false);
  const [tocPosition, setTocPosition] = useState<"center" | "top">("center");
  const [showMobileReadingOptions, setShowMobileReadingOptions] =
    useState(false);

  // Text-to-speech states
  const [isPlaying, setIsPlaying] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);
  const [remainingTime, setRemainingTime] = useState(0);

  // Add a state to track if the screen is small
  const [isSmallScreen, setIsSmallScreen] = useState(false);

  const contentRef = useRef<HTMLDivElement>(null);
  const observerRef = useRef<IntersectionObserver | null>(null);
  const speechReaderRef = useRef<SpeechReader | null>(null);

  // Load reading preferences from localStorage, respecting ThemeProvider's reading mode control
  useEffect(() => {
    const savedPreferences = localStorage.getItem("blog-reading-preferences");
    if (savedPreferences) {
      const preferences = JSON.parse(savedPreferences);
      // Only restore font and line settings, let ThemeProvider handle reading mode
      setFontSize(preferences.fontSize || 18);
      setLineHeight(preferences.lineHeight || 1.6);
    }
  }, []);

  useEffect(() => {
    localStorage.setItem(
      "blog-reading-preferences",
      JSON.stringify({
        fontSize,
        lineHeight,
      })
    );
  }, [fontSize, lineHeight]);

  const handleFontSizeChange = (newSize: number) => {
    setFontSize(Math.max(12, Math.min(24, newSize)));
  };

  const handleLineHeightChange = (newHeight: number) => {
    setLineHeight(Math.max(1.2, Math.min(2.0, newHeight)));
  };

  // Initialize SpeechReader for text-to-speech with highlighting
  const initializeSpeechReader = () => {
    console.log("Initializing SpeechReader...");
    if (!contentRef.current) {
      console.log("No contentRef.current found");
      return null;
    }

    const articleElement = contentRef.current.querySelector("article");
    if (!articleElement) {
      console.log("No article element found");
      return null;
    }

    console.log("Article element found:", articleElement);

    const options: SpeechReaderOptions = {
      highlightColors: {
        wordHighlight: "#6fa8dc",
        paragraphHighlight: "#cfe2f3",
      },
      wordsPerMinute: 200, // Increased for better speed while maintaining clarity
      autoScroll: false, // Disabled to prevent screen jumping during reading
      rate: 0.85, // Optimized rate for speed and clarity balance
      pitch: 1.0,
      volume: 1.0,
    };

    const events: SpeechReaderEvents = {
      onStart: () => {
        setIsPlaying(true);
        setIsPaused(false);
        // Update duration when speech starts
        if (speechReaderRef.current) {
          setDuration(speechReaderRef.current.getTotalDuration());
        }
      },
      onEnd: () => {
        setIsPlaying(false);
        setIsPaused(false);
        setProgress(0);
        setRemainingTime(0);
      },
      onPause: () => {
        setIsPaused(true);
      },
      onResume: () => {
        setIsPaused(false);
      },
      onError: (error) => {
        console.error("Speech Reader Error:", error);
        setIsPlaying(false);
        setIsPaused(false);
        setProgress(0);
        setRemainingTime(0);
      },
      onProgress: (progressPercent) => {
        setProgress(progressPercent);
        // Update remaining time
        if (speechReaderRef.current) {
          setRemainingTime(speechReaderRef.current.getRemainingTime());
        }
      },
    };

    console.log("Creating SpeechReader with article element:", articleElement);
    console.log("SpeechReader options:", options);

    try {
      const speechReader = new SpeechReader(articleElement, options, events);
      console.log("SpeechReader created successfully:", speechReader);
      return speechReader;
    } catch (error) {
      console.error("Error creating SpeechReader:", error);
      return null;
    }
  };

  // Speech control functions using the new SpeechReader
  const startSpeech = () => {
    console.log("startSpeech called");

    // Check for Web Speech API support
    if (!("speechSynthesis" in window)) {
      console.error("Web Speech API not supported");
      alert("Speech synthesis is not supported in your browser.");
      return;
    }

    if (!speechReaderRef.current) {
      console.log("Creating new SpeechReader...");
      speechReaderRef.current = initializeSpeechReader();
    }

    if (speechReaderRef.current) {
      console.log("Starting speech...");
      try {
        speechReaderRef.current.start();
      } catch (error) {
        console.error("Error starting speech:", error);
        const errorMessage =
          error instanceof Error ? error.message : "Unknown error occurred";
        alert(`Error starting speech: ${errorMessage}`);
      }
    } else {
      console.error("Failed to create SpeechReader");
      alert(
        "Failed to initialize speech reader. Please try refreshing the page."
      );
    }
  };

  const pauseSpeech = () => {
    if (speechReaderRef.current) {
      speechReaderRef.current.pause();
    }
  };

  const resumeSpeech = () => {
    if (speechReaderRef.current) {
      speechReaderRef.current.resume();
    }
  };

  const stopSpeech = () => {
    if (speechReaderRef.current) {
      speechReaderRef.current.stop();
      speechReaderRef.current = null;
    }
    // Reset all states to initial values
    setIsPlaying(false);
    setIsPaused(false);
    setProgress(0);
    setDuration(0);
    setRemainingTime(0);
  };

  // Seek to specific position in the audio
  const seekSpeech = (percentage: number) => {
    if (speechReaderRef.current) {
      speechReaderRef.current.seekTo(percentage);
    }
  };

  // Extract headings and create TOC
  const generateToc = useCallback(() => {
    if (!contentRef.current) return;

    const headings = contentRef.current.querySelectorAll(
      "h1, h2, h3, h4, h5, h6"
    );
    const items: TocItem[] = [];

    headings.forEach((heading, index) => {
      const level = parseInt(heading.tagName.charAt(1));
      let id = heading.id;

      // Generate ID if not present
      if (!id) {
        id = `heading-${index}-${
          heading.textContent
            ?.toLowerCase()
            .replace(/[^a-z0-9]+/g, "-")
            .replace(/(^-|-$)/g, "") || ""
        }`;
        heading.id = id;
      }

      items.push({
        id,
        title: heading.textContent || "",
        titleHtml: heading.innerHTML || "",
        level,
        element: heading as HTMLElement,
      });
    });

    setTocItems(items);

    // Set up intersection observer for active section highlighting
    if (observerRef.current) {
      observerRef.current.disconnect();
    }

    observerRef.current = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id);
          }
        });
      },
      {
        rootMargin: "-100px 0px -66%",
        threshold: 0,
      }
    );

    headings.forEach((heading) => {
      observerRef.current?.observe(heading);
    });
  }, []);

  // Smooth scroll to section
  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      const yOffset = -100; // Account for fixed header
      const y =
        element.getBoundingClientRect().top + window.pageYOffset + yOffset;

      window.scrollTo({ top: y, behavior: "smooth" });
    }
  };

  // Generate TOC after content is rendered
  useEffect(() => {
    const timer = setTimeout(() => {
      generateToc();
    }, 100);

    return () => {
      clearTimeout(timer);
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, [generateToc]);

  // Cleanup speech on unmount
  useEffect(() => {
    return () => {
      // Cleanup SpeechReader instance
      if (speechReaderRef.current) {
        speechReaderRef.current.stop();
        speechReaderRef.current = null;
      }
    };
  }, []);

  // Handle scroll to prevent TOC from overlapping footer with smooth animations
  useEffect(() => {
    let animationFrame: number;

    const handleScroll = () => {
      const footer = document.querySelector("footer");
      if (!footer) return;

      const footerRect = footer.getBoundingClientRect();
      const viewportHeight = window.innerHeight;
      const currentScrollY = window.scrollY;

      // TOC dimensions and positioning calculations
      const tocHeight = 500; // Approximate TOC height in pixels
      const safetyMargin = 100; // Safety margin to prevent overlap
      const tocCenterPosition = viewportHeight / 2;
      const tocBottomWhenCentered = tocCenterPosition + tocHeight / 2;

      // Calculate footer visibility and potential overlap
      const footerTopFromViewport = footerRect.top;
      const footerIsVisible = footerTopFromViewport < viewportHeight;
      const footerWouldOverlap =
        footerIsVisible &&
        footerTopFromViewport < tocBottomWhenCentered + safetyMargin;

      // Calculate document proximity to bottom
      const documentHeight = document.documentElement.scrollHeight;
      const distanceFromBottom =
        documentHeight - (currentScrollY + viewportHeight);
      const nearBottomThreshold = 300; // Switch when within 300px of bottom

      // Determine TOC position with hysteresis to prevent flickering
      const shouldSwitchToTop =
        footerWouldOverlap || distanceFromBottom <= nearBottomThreshold;

      // Only update if position actually needs to change
      setTocPosition((prev) => {
        if (prev === "center" && shouldSwitchToTop) {
          return "top";
        } else if (
          prev === "top" &&
          !shouldSwitchToTop &&
          distanceFromBottom > nearBottomThreshold + 50
        ) {
          // Add small buffer to prevent flickering when scrolling back up
          return "center";
        }
        return prev;
      });
    };

    // Use requestAnimationFrame for smoother scroll handling
    const throttledHandleScroll = () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame);
      }
      animationFrame = requestAnimationFrame(handleScroll);
    };

    // Attach event listeners
    window.addEventListener("scroll", throttledHandleScroll, { passive: true });
    window.addEventListener("resize", handleScroll, { passive: true });

    // Initial check
    handleScroll();

    return () => {
      window.removeEventListener("scroll", throttledHandleScroll);
      window.removeEventListener("resize", handleScroll);
      if (animationFrame) {
        cancelAnimationFrame(animationFrame);
      }
    };
  }, []);

  useEffect(() => {
    const handleResize = () => {
      const isSmall = window.innerWidth <= 1024; // Change to lg breakpoint (1024px)
      setIsSmallScreen(isSmall);

      // Auto-hide TOC on mobile/tablet devices
      if (isSmall) {
        setShowToc(false);
      } else {
        setShowToc(true);
      }
    };

    // Initial check
    handleResize();

    // Add event listener for resize
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  return (
    <div className="transition-all duration-300 relative">
      {/* Table of Contents - Fixed Left Sidebar */}
      {tocItems.length > 0 && (
        <div
          className={`fixed left-4 z-40 hidden xl:block transition-all duration-700 ease-out ${
            tocPosition === "center"
              ? "top-1/2 -translate-y-1/2"
              : "top-24 translate-y-0"
          } ${tocCollapsed ? "w-12" : "w-64"}`}
        >
          <div
            className={`p-4 rounded-xl shadow-lg border backdrop-blur-md transition-all duration-300 max-h-[65vh] overflow-y-auto toc-scrollbar ${
              showToc
                ? "opacity-100 translate-x-0"
                : "opacity-0 -translate-x-full"
            }`}
            style={{
              backgroundColor: "var(--background)/95",
              borderColor: "var(--border)",
            }}
          >
            <div className="flex items-center justify-between mb-4">
              <h3
                className={`text-sm font-semibold ${
                  tocCollapsed ? "hidden" : "block"
                }`}
                style={{ color: "var(--text-primary)" }}
              >
                Table of Contents
              </h3>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => setTocCollapsed(!tocCollapsed)}
                  className="p-1 rounded transition-colors duration-200"
                  style={{
                    color: "var(--text-secondary)",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor =
                      "var(--surface-accent)";
                    e.currentTarget.style.color = "var(--accent)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = "transparent";
                    e.currentTarget.style.color = "var(--text-secondary)";
                  }}
                  aria-label={
                    tocCollapsed
                      ? "Expand table of contents"
                      : "Collapse table of contents"
                  }
                >
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d={
                        tocCollapsed
                          ? "M13 5l7 7-7 7M5 5l7 7-7 7" // Expand icon (chevrons pointing right)
                          : "M11 19l-7-7 7-7M19 19l-7-7 7-7" // Collapse icon (chevrons pointing left)
                      }
                    />
                  </svg>
                </button>
              </div>
            </div>

            <nav className={`space-y-1 ${tocCollapsed ? "hidden" : "block"}`}>
              {tocItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => scrollToSection(item.id)}
                  className={`block w-full text-left py-2 px-3 rounded text-sm transition-all duration-200 hover:scale-105 ${
                    activeSection === item.id ? "font-medium" : "font-normal"
                  }`}
                  style={{
                    paddingLeft: `${12 + (item.level - 1) * 12}px`,
                    backgroundColor:
                      activeSection === item.id
                        ? "var(--surface-accent)"
                        : "transparent",
                    color:
                      activeSection === item.id
                        ? "var(--accent)"
                        : "var(--text-secondary)",
                    borderLeft:
                      activeSection === item.id
                        ? `2px solid var(--accent)`
                        : `2px solid transparent`,
                  }}
                  onMouseEnter={(e) => {
                    if (activeSection !== item.id) {
                      e.currentTarget.style.backgroundColor = "var(--surface)";
                      e.currentTarget.style.color = "var(--text-primary)";
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (activeSection !== item.id) {
                      e.currentTarget.style.backgroundColor = "transparent";
                      e.currentTarget.style.color = "var(--text-secondary)";
                    }
                  }}
                  title={item.title}
                >
                  <span
                    className="line-clamp-2 leading-tight toc-item-title"
                    dangerouslySetInnerHTML={{ __html: item.titleHtml }}
                  />
                </button>
              ))}
            </nav>
          </div>
        </div>
      )}

      {/* TOC Toggle Button - Mobile & Tablet */}
      {tocItems.length > 0 && (
        <div className="fixed left-4 top-1/2 -translate-y-1/2 z-40 xl:hidden">
          <button
            onClick={() => setShowToc(!showToc)}
            className="w-10 h-10 rounded-full shadow-lg border transition-all duration-300"
            style={{
              backgroundColor: "var(--background)",
              borderColor: "var(--border)",
              color: "var(--text-secondary)",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = "var(--surface-accent)";
              e.currentTarget.style.color = "var(--accent)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = "var(--background)";
              e.currentTarget.style.color = "var(--text-secondary)";
            }}
            aria-label="Toggle table of contents"
          >
            <svg
              className="w-5 h-5 mx-auto"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M4 6h16M4 12h16M4 18h16"
              />
            </svg>
          </button>
        </div>
      )}

      {/* Mobile TOC Overlay */}
      {tocItems.length > 0 && showToc && (
        <div className="fixed inset-0 z-50 xl:hidden">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => setShowToc(false)}
          />
          <div
            className="absolute left-4 top-1/2 -translate-y-1/2 w-80 max-w-[calc(100vw-2rem)] p-4 rounded-xl shadow-xl border backdrop-blur-md max-h-[70vh] overflow-y-auto toc-scrollbar"
            style={{
              backgroundColor: "var(--background)",
              borderColor: "var(--border)",
            }}
          >
            <div className="flex items-center justify-between mb-4">
              <h3
                className="text-sm font-semibold"
                style={{ color: "var(--text-primary)" }}
              >
                Table of Contents
              </h3>
              <button
                onClick={() => setShowToc(false)}
                className="p-1 rounded transition-colors duration-200"
                style={{
                  color: "var(--text-secondary)",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor =
                    "var(--surface-accent)";
                  e.currentTarget.style.color = "var(--accent)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = "transparent";
                  e.currentTarget.style.color = "var(--text-secondary)";
                }}
                aria-label="Close table of contents"
              >
                <svg
                  className="w-4 h-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>

            <nav className="space-y-1">
              {tocItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => {
                    scrollToSection(item.id);
                    setShowToc(false);
                  }}
                  className={`block w-full text-left py-2 px-3 rounded text-sm transition-all duration-200 ${
                    activeSection === item.id ? "font-medium" : "font-normal"
                  }`}
                  style={{
                    paddingLeft: `${12 + (item.level - 1) * 12}px`,
                    backgroundColor:
                      activeSection === item.id
                        ? "var(--surface-accent)"
                        : "transparent",
                    color:
                      activeSection === item.id
                        ? "var(--accent)"
                        : "var(--text-secondary)",
                    borderLeft:
                      activeSection === item.id
                        ? `2px solid var(--accent)`
                        : `2px solid transparent`,
                  }}
                  onMouseEnter={(e) => {
                    if (activeSection !== item.id) {
                      e.currentTarget.style.backgroundColor = "var(--surface)";
                      e.currentTarget.style.color = "var(--text-primary)";
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (activeSection !== item.id) {
                      e.currentTarget.style.backgroundColor = "transparent";
                      e.currentTarget.style.color = "var(--text-secondary)";
                    }
                  }}
                  title={item.title}
                >
                  <span
                    className="leading-tight toc-item-title"
                    dangerouslySetInnerHTML={{ __html: item.titleHtml }}
                  />
                </button>
              ))}
            </nav>
          </div>
        </div>
      )}

      {/* Reading Controls - Fixed Position */}
      <div className="fixed top-20 right-4 z-50 hidden lg:block">
        <div
          className="p-4 rounded-xl shadow-lg border backdrop-blur-md"
          style={{
            backgroundColor: "var(--background)/95",
            borderColor: "var(--border)",
          }}
        >
          <h3
            className="text-sm font-semibold mb-3"
            style={{ color: "var(--text-primary)" }}
          >
            Reading Options
          </h3>

          {/* Reading Mode Toggle */}
          <div className="flex items-center justify-between mb-3">
            <span
              className="text-xs"
              style={{ color: "var(--text-secondary)" }}
            >
              Eye Comfort
            </span>
            <button
              onClick={() => setReadingMode(!isReadingMode)}
              className={`relative w-10 h-5 rounded-full transition-colors duration-200 ${
                isReadingMode ? "bg-amber-500" : "bg-gray-300 dark:bg-gray-600"
              }`}
              aria-label="Toggle reading mode"
            >
              <div
                className={`absolute top-0.5 w-4 h-4 bg-white rounded-full transition-transform duration-200 ${
                  isReadingMode ? "translate-x-5" : "translate-x-0.5"
                }`}
              />
            </button>
          </div>

          {/* Font Size Control */}
          <div className="mb-3">
            <div className="flex items-center justify-between mb-1">
              <span
                className="text-xs"
                style={{ color: "var(--text-secondary)" }}
              >
                Font Size
              </span>
              <span
                className="text-xs font-mono"
                style={{ color: "var(--text-primary)" }}
              >
                {fontSize}px
              </span>
            </div>
            <div className="flex items-center gap-1">
              <button
                onClick={() => handleFontSizeChange(fontSize - 2)}
                className="w-6 h-6 rounded text-xs font-bold transition-colors duration-200"
                style={{
                  backgroundColor: "var(--surface)",
                  color: "var(--text-secondary)",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor =
                    "var(--surface-accent)";
                  e.currentTarget.style.color = "var(--accent)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = "var(--surface)";
                  e.currentTarget.style.color = "var(--text-secondary)";
                }}
                aria-label="Decrease font size"
              >
                A-
              </button>
              <div className="flex-1 mx-2">
                <input
                  type="range"
                  min="12"
                  max="24"
                  value={fontSize}
                  onChange={(e) => handleFontSizeChange(Number(e.target.value))}
                  className="w-full h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                  style={{
                    background: `linear-gradient(to right, var(--accent) 0%, var(--accent) ${
                      ((fontSize - 12) / (24 - 12)) * 100
                    }%, var(--border) ${
                      ((fontSize - 12) / (24 - 12)) * 100
                    }%, var(--border) 100%)`,
                  }}
                />
              </div>
              <button
                onClick={() => handleFontSizeChange(fontSize + 2)}
                className="w-6 h-6 rounded text-xs font-bold transition-colors duration-200"
                style={{
                  backgroundColor: "var(--surface)",
                  color: "var(--text-secondary)",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor =
                    "var(--surface-accent)";
                  e.currentTarget.style.color = "var(--accent)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = "var(--surface)";
                  e.currentTarget.style.color = "var(--text-secondary)";
                }}
                aria-label="Increase font size"
              >
                A+
              </button>
            </div>
          </div>

          {/* Line Height Control */}
          <div className="mb-4">
            <div className="flex items-center justify-between mb-1">
              <span
                className="text-xs"
                style={{ color: "var(--text-secondary)" }}
              >
                Line Spacing
              </span>
              <span
                className="text-xs font-mono"
                style={{ color: "var(--text-primary)" }}
              >
                {lineHeight.toFixed(1)}
              </span>
            </div>
            <div className="flex items-center gap-1">
              <button
                onClick={() => handleLineHeightChange(lineHeight - 0.1)}
                className="w-6 h-6 rounded text-xs font-bold transition-colors duration-200"
                style={{
                  backgroundColor: "var(--surface)",
                  color: "var(--text-secondary)",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor =
                    "var(--surface-accent)";
                  e.currentTarget.style.color = "var(--accent)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = "var(--surface)";
                  e.currentTarget.style.color = "var(--text-secondary)";
                }}
                aria-label="Decrease line height"
              >
                ≡
              </button>
              <div className="flex-1 mx-2">
                <input
                  type="range"
                  min="1.2"
                  max="2.0"
                  step="0.1"
                  value={lineHeight}
                  onChange={(e) =>
                    handleLineHeightChange(Number(e.target.value))
                  }
                  className="w-full h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                  style={{
                    background: `linear-gradient(to right, var(--accent) 0%, var(--accent) ${
                      ((lineHeight - 1.2) / (2.0 - 1.2)) * 100
                    }%, var(--border) ${
                      ((lineHeight - 1.2) / (2.0 - 1.2)) * 100
                    }%, var(--border) 100%)`,
                  }}
                />
              </div>
              <button
                onClick={() => handleLineHeightChange(lineHeight + 0.1)}
                className="w-6 h-6 rounded text-xs font-bold transition-colors duration-200"
                style={{
                  backgroundColor: "var(--surface)",
                  color: "var(--text-secondary)",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor =
                    "var(--surface-accent)";
                  e.currentTarget.style.color = "var(--accent)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = "var(--surface)";
                  e.currentTarget.style.color = "var(--text-secondary)";
                }}
                aria-label="Increase line height"
              >
                ☰
              </button>
            </div>
          </div>

          {/* Text-to-Speech Controls */}
          <div
            className="border-t pt-3"
            style={{ borderColor: "var(--border)" }}
          >
            <h4
              className="text-xs font-semibold mb-3"
              style={{ color: "var(--text-primary)" }}
            >
              Audio Reading
            </h4>

            {/* Play/Pause/Stop Controls */}
            <div className="flex items-center gap-2 mb-3">
              {!isPlaying ? (
                // Show Play button when not playing
                <button
                  onClick={startSpeech}
                  className="flex-1 px-3 py-2 rounded-lg transition-all duration-200 flex items-center justify-center gap-2"
                  style={{
                    backgroundColor: "var(--surface)",
                    color: "var(--text-primary)",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor =
                      "var(--surface-accent)";
                    e.currentTarget.style.color = "var(--accent)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = "var(--surface)";
                    e.currentTarget.style.color = "var(--text-primary)";
                  }}
                  aria-label="Start reading"
                >
                  <svg
                    className="w-4 h-4"
                    fill="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path d="M8 5v14l11-7z" />
                  </svg>
                  <span className="text-xs font-medium">Play</span>
                </button>
              ) : (
                // Show Pause/Continue and Stop buttons when playing
                <>
                  <button
                    onClick={isPaused ? resumeSpeech : pauseSpeech}
                    className="flex-1 px-3 py-2 rounded-lg transition-all duration-200 flex items-center justify-center gap-2"
                    style={{
                      backgroundColor: isPaused
                        ? "var(--accent)"
                        : "var(--accent)",
                      color: "white",
                    }}
                    aria-label={isPaused ? "Resume reading" : "Pause reading"}
                  >
                    {isPaused ? (
                      <svg
                        className="w-4 h-4"
                        fill="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path d="M8 5v14l11-7z" />
                      </svg>
                    ) : (
                      <svg
                        className="w-4 h-4"
                        fill="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
                      </svg>
                    )}
                    <span className="text-xs font-medium">
                      {isPaused ? "Continue" : "Pause"}
                    </span>
                  </button>

                  <button
                    onClick={stopSpeech}
                    className="px-3 py-2 rounded-lg transition-all duration-200 flex items-center justify-center gap-1"
                    style={{
                      backgroundColor: "var(--surface)",
                      color: "var(--text-secondary)",
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor =
                        "var(--surface-accent)";
                      e.currentTarget.style.color = "var(--accent)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = "var(--surface)";
                      e.currentTarget.style.color = "var(--text-secondary)";
                    }}
                    aria-label="Stop reading"
                  >
                    <svg
                      className="w-4 h-4"
                      fill="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path d="M6 6h12v12H6z" />
                    </svg>
                    <span className="text-xs font-medium">Stop</span>
                  </button>
                </>
              )}
            </div>

            {/* Progress Bar */}
            {isPlaying && (
              <div className="mb-3">
                <div className="flex items-center justify-between mb-1">
                  <span
                    className="text-xs"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Progress
                  </span>
                  <div className="flex items-center gap-2">
                    <span
                      className="text-xs font-mono"
                      style={{ color: "var(--text-primary)" }}
                    >
                      {Math.floor(remainingTime / 60)}:
                      {Math.floor(remainingTime % 60)
                        .toString()
                        .padStart(2, "0")}
                    </span>
                    <span
                      className="text-xs"
                      style={{ color: "var(--text-primary)" }}
                    >
                      {Math.round(progress)}%
                    </span>
                  </div>
                </div>
                <div
                  className="w-full h-2 rounded-full cursor-pointer relative group"
                  style={{ backgroundColor: "var(--border)" }}
                  onClick={(e) => {
                    const rect = e.currentTarget.getBoundingClientRect();
                    const clickX = e.clientX - rect.left;
                    const percentage = (clickX / rect.width) * 100;
                    seekSpeech(Math.max(0, Math.min(100, percentage)));
                  }}
                >
                  <div
                    className="h-2 rounded-full transition-all duration-300"
                    style={{
                      backgroundColor: "var(--accent)",
                      width: `${progress}%`,
                    }}
                  />
                  {/* Hover indicator */}
                  <div
                    className="absolute top-0 w-full h-2 rounded-full opacity-0 group-hover:opacity-20 transition-opacity duration-200"
                    style={{ backgroundColor: "var(--accent)" }}
                  />
                </div>
                <div className="flex justify-between mt-1">
                  <span
                    className="text-xs"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    {Math.floor((duration - remainingTime) / 60)}:
                    {Math.floor((duration - remainingTime) % 60)
                      .toString()
                      .padStart(2, "0")}
                  </span>
                  <span
                    className="text-xs"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    {Math.floor(duration / 60)}:
                    {Math.floor(duration % 60)
                      .toString()
                      .padStart(2, "0")}
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Main Content - Always Centered */}
      <div className="relative z-10 w-full">
        <div className="flex justify-center items-start py-12">
          <div
            className="w-full max-w-4xl px-6 mb-20"
            ref={contentRef}
            style={{
              margin: "0 auto",
              position: "relative",
            }}
          >
            {/* Article Header */}
            <header className="mb-12 text-center">
              {/* Collection tag */}
              {collection && (
                <div className="mb-6">
                  <CollectionTag collection={collection} variant="detailed" />
                </div>
              )}

              <h1
                className="text-lg md:text-2xl lg:text-3xl xl:text-4xl font-bold mb-8 leading-tight"
                style={{
                  color: isReadingMode
                    ? theme === "dark"
                      ? "#f5e6d3" // Warm off-white for dark mode
                      : "#92400e"
                    : "var(--text-primary)",
                  fontSize: `clamp(1.5rem, ${fontSize * 1.8}px, 2.5rem)`,
                  lineHeight: lineHeight * 0.9,
                }}
              >
                {title}
              </h1>

              {/* Article Meta */}
              <div className="flex flex-wrap items-center justify-center gap-4 mb-6">
                <div className="flex items-center gap-2">
                  <div
                    className="w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-bold"
                    style={{ backgroundColor: "var(--accent)" }}
                  >
                    {author.charAt(0)}
                  </div>
                  <span
                    className="text-sm font-medium"
                    style={{
                      color: isReadingMode
                        ? theme === "dark"
                          ? "#e8d5b7" // Slightly darker warm color for author
                          : "#92400e"
                        : "var(--text-primary)",
                    }}
                  >
                    {author}
                  </span>
                </div>
                <div className="flex items-center gap-1">
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    style={{
                      color: isReadingMode
                        ? theme === "dark"
                          ? "#d97706"
                          : "#78350f"
                        : "var(--text-secondary)",
                    }}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
                    />
                  </svg>
                  <time
                    className="text-sm"
                    style={{
                      color: isReadingMode
                        ? theme === "dark"
                          ? "#d97706"
                          : "#78350f"
                        : "var(--text-secondary)",
                    }}
                  >
                    {formatDate(publishDate)}
                  </time>
                </div>{" "}
                <div className="flex items-center gap-1">
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    style={{
                      color: isReadingMode
                        ? theme === "dark"
                          ? "#d97706"
                          : "#78350f"
                        : "var(--text-secondary)",
                    }}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                  <span
                    className="text-sm"
                    style={{
                      color: isReadingMode
                        ? theme === "dark"
                          ? "#d97706"
                          : "#78350f"
                        : "var(--text-secondary)",
                    }}
                  >
                    {readTime}
                  </span>
                </div>
                <span
                  className="px-3 py-1 rounded-full text-xs font-medium"
                  style={{
                    backgroundColor: isReadingMode
                      ? theme === "dark"
                        ? "#52403d"
                        : "#fef3c7"
                      : "var(--surface-accent)",
                    color: isReadingMode
                      ? theme === "dark"
                        ? "#fbbf24"
                        : "#92400e"
                      : "var(--accent)",
                  }}
                >
                  {category}
                </span>
              </div>

              {/* Tags */}
              {tags.length > 0 && (
                <div className="flex flex-wrap gap-2 mb-6">
                  {tags.map((tag, index) => (
                    <span
                      key={index}
                      className="px-2 py-1 rounded text-xs font-medium border transition-colors duration-200"
                      style={{
                        backgroundColor: isReadingMode
                          ? theme === "dark"
                            ? "rgba(82, 64, 61, 0.3)"
                            : "rgba(254, 243, 199, 0.5)"
                          : "var(--surface)",
                        borderColor: isReadingMode
                          ? theme === "dark"
                            ? "#52403d"
                            : "#f3e8ff"
                          : "var(--border)",
                        color: isReadingMode
                          ? theme === "dark"
                            ? "#fbbf24"
                            : "#92400e"
                          : "var(--text-secondary)",
                      }}
                    >
                      #{tag}
                    </span>
                  ))}
                </div>
              )}
            </header>

            {/* Article Content */}
            <article
              className="prose prose-xl max-w-none mx-auto mb-16"
              style={{
                fontSize: `${fontSize}px`,
                lineHeight: lineHeight,
                color: isReadingMode
                  ? theme === "dark"
                    ? "#f5e6d3"
                    : "#92400e"
                  : "var(--text-primary)",
              }}
            >
              <MathJax
                className="blog-content"
                style={
                  {
                    "--reading-text-color": isReadingMode
                      ? theme === "dark"
                        ? "#f5e6d3"
                        : "#92400e"
                      : "var(--text-primary)",
                    "--reading-text-secondary": isReadingMode
                      ? theme === "dark"
                        ? "#e8d5b7"
                        : "#78350f"
                      : "var(--text-secondary)",
                    "--reading-accent": isReadingMode
                      ? theme === "dark"
                        ? "#fbbf24"
                        : "#b45309"
                      : "var(--accent)",
                    "--reading-border": isReadingMode
                      ? theme === "dark"
                        ? "#52403d"
                        : "#fef3c7"
                      : "var(--border)",
                    "--reading-surface": isReadingMode
                      ? theme === "dark"
                        ? "rgba(82, 64, 61, 0.2)"
                        : "rgba(254, 243, 199, 0.3)"
                      : "var(--surface)",
                  } as React.CSSProperties
                }
              >
                {dangerouslySetInnerHTML ? (
                  <div dangerouslySetInnerHTML={dangerouslySetInnerHTML} />
                ) : (
                  children
                )}
              </MathJax>
            </article>

            {/* Blog Share Section */}
            {postSlug && <BlogShareSection postSlug={postSlug} title={title} />}
          </div>
        </div>
      </div>

      {/* Mobile/Tablet Reading Controls (show on <lg screens) */}
      <div className="lg:hidden fixed bottom-20 right-4 z-50">
        {/* Reading Options Toggle Button */}
        <button
          onClick={() => setShowMobileReadingOptions(!showMobileReadingOptions)}
          className="w-12 h-12 rounded-full shadow-lg border backdrop-blur-md transition-all duration-200 flex items-center justify-center mb-2"
          style={{
            backgroundColor: showMobileReadingOptions
              ? "var(--accent)"
              : "var(--background)/95",
            borderColor: "var(--border)",
            color: showMobileReadingOptions ? "white" : "var(--text-primary)",
          }}
          aria-label="Toggle reading options"
        >
          <svg
            className="w-5 h-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4"
            />
          </svg>
        </button>

        {/* Reading Options Dropdown */}
        {showMobileReadingOptions && (
          <div
            className="p-4 rounded-xl shadow-lg border backdrop-blur-md w-72"
            style={{
              backgroundColor: "var(--background)/95",
              borderColor: "var(--border)",
            }}
          >
            {/* Eye Comfort Toggle */}
            <div className="flex items-center justify-between mb-4">
              <span
                className="text-sm font-medium"
                style={{ color: "var(--text-primary)" }}
              >
                Eye Comfort
              </span>
              <button
                onClick={() => setReadingMode(!isReadingMode)}
                className={`relative w-11 h-6 rounded-full transition-colors duration-200 ${
                  isReadingMode
                    ? "bg-amber-500"
                    : "bg-gray-300 dark:bg-gray-600"
                }`}
                aria-label="Toggle reading mode"
              >
                <div
                  className={`absolute top-0.5 w-5 h-5 bg-white rounded-full transition-transform duration-200 ${
                    isReadingMode ? "translate-x-5" : "translate-x-0.5"
                  }`}
                />
              </button>
            </div>

            {/* Font Size Control */}
            <div className="mb-4">
              <div className="flex items-center justify-between mb-2">
                <span
                  className="text-sm font-medium"
                  style={{ color: "var(--text-primary)" }}
                >
                  Font Size
                </span>
                <span
                  className="text-xs"
                  style={{ color: "var(--text-secondary)" }}
                >
                  {fontSize}px
                </span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleFontSizeChange(fontSize - 1)}
                  className="w-7 h-7 rounded text-xs font-bold transition-colors duration-200 flex items-center justify-center"
                  style={{
                    backgroundColor: "var(--surface)",
                    color: "var(--text-secondary)",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor =
                      "var(--surface-accent)";
                    e.currentTarget.style.color = "var(--accent)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = "var(--surface)";
                    e.currentTarget.style.color = "var(--text-secondary)";
                  }}
                  aria-label="Decrease font size"
                >
                  A-
                </button>
                <div className="flex-1 mx-2">
                  <input
                    type="range"
                    min="12"
                    max="24"
                    step="1"
                    value={fontSize}
                    onChange={(e) =>
                      handleFontSizeChange(Number(e.target.value))
                    }
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                    style={{
                      background: `linear-gradient(to right, var(--accent) 0%, var(--accent) ${
                        ((fontSize - 12) / (24 - 12)) * 100
                      }%, var(--border) ${
                        ((fontSize - 12) / (24 - 12)) * 100
                      }%, var(--border) 100%)`,
                    }}
                  />
                </div>
                <button
                  onClick={() => handleFontSizeChange(fontSize + 1)}
                  className="w-7 h-7 rounded text-xs font-bold transition-colors duration-200 flex items-center justify-center"
                  style={{
                    backgroundColor: "var(--surface)",
                    color: "var(--text-secondary)",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor =
                      "var(--surface-accent)";
                    e.currentTarget.style.color = "var(--accent)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = "var(--surface)";
                    e.currentTarget.style.color = "var(--text-secondary)";
                  }}
                  aria-label="Increase font size"
                >
                  A+
                </button>
              </div>
            </div>

            {/* Line Height Control */}
            <div className="mb-4">
              <div className="flex items-center justify-between mb-2">
                <span
                  className="text-sm font-medium"
                  style={{ color: "var(--text-primary)" }}
                >
                  Line Height
                </span>
                <span
                  className="text-xs"
                  style={{ color: "var(--text-secondary)" }}
                >
                  {lineHeight.toFixed(1)}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleLineHeightChange(lineHeight - 0.1)}
                  className="w-7 h-7 rounded text-xs font-bold transition-colors duration-200 flex items-center justify-center"
                  style={{
                    backgroundColor: "var(--surface)",
                    color: "var(--text-secondary)",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor =
                      "var(--surface-accent)";
                    e.currentTarget.style.color = "var(--accent)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = "var(--surface)";
                    e.currentTarget.style.color = "var(--text-secondary)";
                  }}
                  aria-label="Decrease line height"
                >
                  ≡
                </button>
                <div className="flex-1 mx-2">
                  <input
                    type="range"
                    min="1.2"
                    max="2.0"
                    step="0.1"
                    value={lineHeight}
                    onChange={(e) =>
                      handleLineHeightChange(Number(e.target.value))
                    }
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                    style={{
                      background: `linear-gradient(to right, var(--accent) 0%, var(--accent) ${
                        ((lineHeight - 1.2) / (2.0 - 1.2)) * 100
                      }%, var(--border) ${
                        ((lineHeight - 1.2) / (2.0 - 1.2)) * 100
                      }%, var(--border) 100%)`,
                    }}
                  />
                </div>
                <button
                  onClick={() => handleLineHeightChange(lineHeight + 0.1)}
                  className="w-7 h-7 rounded text-xs font-bold transition-colors duration-200 flex items-center justify-center"
                  style={{
                    backgroundColor: "var(--surface)",
                    color: "var(--text-secondary)",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor =
                      "var(--surface-accent)";
                    e.currentTarget.style.color = "var(--accent)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = "var(--surface)";
                    e.currentTarget.style.color = "var(--text-secondary)";
                  }}
                  aria-label="Increase line height"
                >
                  ≡
                </button>
              </div>
            </div>

            {/* Audio Reading Controls */}
            <div
              className="border-t pt-4"
              style={{ borderColor: "var(--border)" }}
            >
              <div className="flex items-center justify-between mb-3">
                <span
                  className="text-sm font-medium"
                  style={{ color: "var(--text-primary)" }}
                >
                  Audio Reading
                </span>
                {!isPlaying ? (
                  <button
                    onClick={startSpeech}
                    className="px-3 py-1.5 rounded-lg transition-all duration-200 text-sm flex items-center gap-2"
                    style={{
                      backgroundColor: "var(--surface)",
                      color: "var(--text-primary)",
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor =
                        "var(--surface-accent)";
                      e.currentTarget.style.color = "var(--accent)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = "var(--surface)";
                      e.currentTarget.style.color = "var(--text-primary)";
                    }}
                    aria-label="Start reading"
                  >
                    <svg
                      className="w-4 h-4"
                      fill="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path d="M8 5v14l11-7z" />
                    </svg>
                    Play
                  </button>
                ) : (
                  <div className="flex items-center gap-2">
                    <button
                      onClick={isPaused ? resumeSpeech : pauseSpeech}
                      className="px-3 py-1.5 rounded-lg transition-all duration-200 text-sm flex items-center gap-2"
                      style={{
                        backgroundColor: "var(--accent)",
                        color: "white",
                      }}
                      aria-label={isPaused ? "Resume reading" : "Pause reading"}
                    >
                      {isPaused ? (
                        <svg
                          className="w-4 h-4"
                          fill="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path d="M8 5v14l11-7z" />
                        </svg>
                      ) : (
                        <svg
                          className="w-4 h-4"
                          fill="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
                        </svg>
                      )}
                      {isPaused ? "Continue" : "Pause"}
                    </button>

                    <button
                      onClick={stopSpeech}
                      className="px-2 py-1.5 rounded-lg transition-all duration-200 text-sm"
                      style={{
                        backgroundColor: "var(--surface)",
                        color: "var(--text-secondary)",
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.backgroundColor =
                          "var(--surface-accent)";
                        e.currentTarget.style.color = "var(--accent)";
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.backgroundColor =
                          "var(--surface)";
                        e.currentTarget.style.color = "var(--text-secondary)";
                      }}
                      aria-label="Stop reading"
                    >
                      <svg
                        className="w-4 h-4"
                        fill="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path d="M6 6h12v12H6z" />
                      </svg>
                    </button>
                  </div>
                )}
              </div>

              {/* Show progress bar when playing */}
              {isPlaying && (
                <div className="mt-3">
                  <div className="flex items-center justify-between mb-2">
                    <span
                      className="text-xs"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      Progress
                    </span>
                    <div className="flex items-center gap-2">
                      <span
                        className="text-xs font-mono"
                        style={{ color: "var(--text-primary)" }}
                      >
                        {Math.floor(remainingTime / 60)}:
                        {Math.floor(remainingTime % 60)
                          .toString()
                          .padStart(2, "0")}
                      </span>
                      <span
                        className="text-xs"
                        style={{ color: "var(--text-primary)" }}
                      >
                        {Math.round(progress)}%
                      </span>
                    </div>
                  </div>
                  <div
                    className="w-full h-2 rounded-full mb-2 cursor-pointer relative group"
                    style={{ backgroundColor: "var(--border)" }}
                    onClick={(e) => {
                      const rect = e.currentTarget.getBoundingClientRect();
                      const clickX = e.clientX - rect.left;
                      const percentage = (clickX / rect.width) * 100;
                      seekSpeech(Math.max(0, Math.min(100, percentage)));
                    }}
                  >
                    <div
                      className="h-2 rounded-full transition-all duration-300"
                      style={{
                        backgroundColor: "var(--accent)",
                        width: `${progress}%`,
                      }}
                    />
                    {/* Hover indicator */}
                    <div
                      className="absolute top-0 w-full h-2 rounded-full opacity-0 group-hover:opacity-20 transition-opacity duration-200"
                      style={{ backgroundColor: "var(--accent)" }}
                    />
                  </div>
                  <div className="flex justify-between items-center">
                    <div
                      className="flex justify-between text-xs w-full"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      <span>
                        {Math.floor((duration - remainingTime) / 60)}:
                        {Math.floor((duration - remainingTime) % 60)
                          .toString()
                          .padStart(2, "0")}
                      </span>
                      <span>
                        {Math.floor(duration / 60)}:
                        {Math.floor(duration % 60)
                          .toString()
                          .padStart(2, "0")}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Prevent duplicate TOC on small screens */}
      {tocItems.length > 0 && !isSmallScreen && showToc && (
        <div className="toc-container">{/* Render TOC */}</div>
      )}
    </div>
  );
}
