"use client";

import React, { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { useTheme } from "../ThemeProvider";
import CollectionTag from "../../components/CollectionTag";
import AiGeneratedBadge from "../../components/AiGeneratedBadge";
import { TagList } from "../../components/TagBadge";
import "katex/dist/katex.min.css";
import MathJax from "./MathJax";
import "./BlogContent.css"; // Re-enabled for proper styling
import "../../components/styles/TextHighlight.css"; // Import text highlighting styles
import type {
  SpeechReader,
  SpeechReaderOptions,
  SpeechReaderEvents,
} from "../../components/utils/SpeechReader";
import { formatDate } from "../../lib/dateUtils";
import BlogGraphSidebar from "./BlogGraphSidebar";
import BlogHighlighter from "./highlights/BlogHighlighter";

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
  publishDate?: string;
  readTime?: string;
  tags?: string[];
  category?: string;
  author?: string;
  postSlug?: string;
  collection?: string;
  aiGenerated?: boolean;
  dangerouslySetInnerHTML?: { __html: string };
}

export default function BlogReader({
  children,
  title,
  publishDate,
  readTime,
  tags = [],
  category,
  author,
  postSlug,
  collection,
  aiGenerated,
  dangerouslySetInnerHTML,
}: BlogReaderProps) {
  const { theme, isReadingMode, toggleReadingMode } = useTheme();
  const [tocItems, setTocItems] = useState<TocItem[]>([]);
  const [activeSection, setActiveSection] = useState<string>("");
  const [showToc, setShowToc] = useState(false);
  const [tocCollapsed, setTocCollapsed] = useState(false);
  const [tocPosition, setTocPosition] = useState<"center" | "top">("center");
  const [sidebarBottomOffset, setSidebarBottomOffset] = useState<number>(0);

  // Text-to-speech states
  const [isPlaying, setIsPlaying] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);
  const [remainingTime, setRemainingTime] = useState(0);

  // Add a state to track if the screen is small

  // Image lightbox state
  const [lightboxImage, setLightboxImage] = useState<string | null>(null);
  const [lightboxAlt, setLightboxAlt] = useState<string>("");

  const contentRef = useRef<HTMLDivElement>(null);
  const articleRef = useRef<HTMLElement | null>(null);
  const speechReaderRef = useRef<SpeechReader | null>(null);

  // Memoize reading-mode colors to avoid recomputing on every render
  const readingColors = useMemo(() => {
    if (!isReadingMode) {
      return {
        textPrimary: "var(--text-primary)",
        textSecondary: "var(--text-secondary)",
        accent: "var(--accent)",
        border: "var(--border)",
        surface: "var(--surface)",
        surfaceAccent: "var(--surface-accent)",
      };
    }
    const isDark = theme === "dark";
    return {
      textPrimary: isDark ? "#f5e6d3" : "#92400e",
      textSecondary: isDark ? "#d97706" : "#78350f",
      accent: isDark ? "#fbbf24" : "#b45309",
      border: isDark ? "#52403d" : "#fef3c7",
      surface: isDark ? "rgba(82, 64, 61, 0.2)" : "rgba(254, 243, 199, 0.3)",
      surfaceAccent: isDark ? "#52403d" : "#fef3c7",
    };
  }, [isReadingMode, theme]);

  // Memoize reading-mode CSS variables for MathJax
  const readingCssVars = useMemo(
    () =>
      ({
        "--reading-text-color": readingColors.textPrimary,
        "--reading-text-secondary": isReadingMode
          ? theme === "dark"
            ? "#e8d5b7"
            : "#78350f"
          : "var(--text-secondary)",
        "--reading-accent": readingColors.accent,
        "--reading-border": readingColors.border,
        "--reading-surface": readingColors.surface,
      }) as React.CSSProperties,
    [readingColors, isReadingMode, theme],
  );

  // Initialize SpeechReader for text-to-speech with highlighting (lazy-loaded)
  const initializeSpeechReader = useCallback(async (): Promise<SpeechReader | null> => {
    if (!contentRef.current) {
      return null;
    }

    const articleElement = contentRef.current.querySelector("article");
    if (!articleElement) {
      return null;
    }

    const options: SpeechReaderOptions = {
      highlightColors: {
        wordHighlight: "#6fa8dc",
        paragraphHighlight: "#cfe2f3",
      },
      wordsPerMinute: 200,
      autoScroll: false,
      rate: 0.85,
      pitch: 1.0,
      volume: 1.0,
    };

    const events: SpeechReaderEvents = {
      onStart: () => {
        setIsPlaying(true);
        setIsPaused(false);
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
        if (speechReaderRef.current) {
          setRemainingTime(speechReaderRef.current.getRemainingTime());
        }
      },
    };

    try {
      const { SpeechReader: SpeechReaderClass } = await import(
        "../../components/utils/SpeechReader"
      );
      const speechReader = new SpeechReaderClass(articleElement, options, events);
      return speechReader;
    } catch (error) {
      console.error("Error creating SpeechReader:", error);
      return null;
    }
  }, []);

  // Speech control functions - memoized to prevent BlogGraphSidebar re-renders
  const startSpeech = useCallback(async () => {
    if (!("speechSynthesis" in window)) {
      alert("Speech synthesis is not supported in your browser.");
      return;
    }

    if (!speechReaderRef.current) {
      speechReaderRef.current = await initializeSpeechReader();
    }

    if (speechReaderRef.current) {
      try {
        speechReaderRef.current.start();
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : "Unknown error occurred";
        alert(`Error starting speech: ${errorMessage}`);
      }
    } else {
      alert(
        "Failed to initialize speech reader. Please try refreshing the page.",
      );
    }
  }, [initializeSpeechReader]);

  const pauseSpeech = useCallback(() => {
    speechReaderRef.current?.pause();
  }, []);

  const resumeSpeech = useCallback(() => {
    speechReaderRef.current?.resume();
  }, []);

  const stopSpeech = useCallback(() => {
    if (speechReaderRef.current) {
      speechReaderRef.current.stop();
      speechReaderRef.current = null;
    }
    setIsPlaying(false);
    setIsPaused(false);
    setProgress(0);
    setDuration(0);
    setRemainingTime(0);
  }, []);

  const seekSpeech = useCallback((percentage: number) => {
    speechReaderRef.current?.seekTo(percentage);
  }, []);

  // Extract headings and create TOC
  const generateToc = useCallback(() => {
    if (!contentRef.current) return;

    const headings = contentRef.current.querySelectorAll(
      "h1, h2, h3, h4, h5, h6",
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
  }, []);

  // Update active section based on scroll position - throttled with rAF
  useEffect(() => {
    if (tocItems.length === 0) return;

    let rafId: number;

    const handleScroll = () => {
      cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(() => {
        const viewportMiddle = window.scrollY + window.innerHeight / 2;

        let closestId = tocItems[0]?.id || "";
        let closestDistance = Infinity;

        for (let i = 0; i < tocItems.length; i++) {
          const el = tocItems[i].element;
          if (el) {
            const rect = el.getBoundingClientRect();
            const distance = Math.abs(
              viewportMiddle - (window.scrollY + rect.top + rect.height / 2),
            );
            if (distance < closestDistance) {
              closestDistance = distance;
              closestId = tocItems[i].id;
            }
          }
        }

        setActiveSection((prev) => {
          if (prev !== closestId) {
            const activeButton = document.querySelector(
              `[data-toc-id="${closestId}"]`,
            );
            activeButton?.scrollIntoView({
              behavior: "smooth",
              block: "nearest",
            });
          }
          return closestId;
        });
      });
    };

    handleScroll();
    window.addEventListener("scroll", handleScroll, { passive: true });

    return () => {
      window.removeEventListener("scroll", handleScroll);
      cancelAnimationFrame(rafId);
    };
  }, [tocItems]);

  // Smooth scroll to section - memoized
  const scrollToSection = useCallback((id: string) => {
    const element = document.getElementById(id);
    if (element) {
      const yOffset = -100; // Account for fixed header
      const y =
        element.getBoundingClientRect().top + window.pageYOffset + yOffset;

      window.scrollTo({ top: y, behavior: "smooth" });
    }
  }, []);

  // Generate TOC after content is rendered
  useEffect(() => {
    const timer = setTimeout(() => {
      generateToc();
    }, 100);

    return () => {
      clearTimeout(timer);
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

      // Calculate bottom offset when footer is visible to keep sidebars above footer
      let bottomOffset = 0;
      if (footerIsVisible && footerTopFromViewport < viewportHeight) {
        // Calculate how much the sidebar needs to move up to stay above footer
        bottomOffset = Math.max(0, viewportHeight - footerTopFromViewport + 20); // 20px extra margin
      }
      setSidebarBottomOffset(bottomOffset);

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

    // Consolidated resize handler - handles both footer overlap and TOC visibility
    const handleResize = () => {
      handleScroll();
      setShowToc(window.innerWidth > 1024);
    };

    window.addEventListener("scroll", throttledHandleScroll, { passive: true });
    window.addEventListener("resize", handleResize, { passive: true });

    // Initial check
    handleScroll();
    setShowToc(window.innerWidth > 1024);

    return () => {
      window.removeEventListener("scroll", throttledHandleScroll);
      window.removeEventListener("resize", handleResize);
      if (animationFrame) {
        cancelAnimationFrame(animationFrame);
      }
    };
  }, []);

  // Image lightbox - attach click handlers to images in content
  useEffect(() => {
    if (!contentRef.current) return;

    const handleImageClick = (e: Event) => {
      const target = e.target as HTMLImageElement;
      if (target.tagName === "IMG" && target.src) {
        setLightboxImage(target.src);
        setLightboxAlt(target.alt || "");
      }
    };

    const images = contentRef.current.querySelectorAll("article img");
    images.forEach((img) => {
      (img as HTMLElement).style.cursor = "zoom-in";
      img.addEventListener("click", handleImageClick);
    });

    return () => {
      images.forEach((img) => {
        img.removeEventListener("click", handleImageClick);
      });
    };
  }, [dangerouslySetInnerHTML, children]);

  // Close lightbox on Escape key - only attach when lightbox is open
  useEffect(() => {
    if (!lightboxImage) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setLightboxImage(null);
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [lightboxImage]);

  return (
    <div className="transition-all duration-300 relative overflow-x-hidden">
      {/* Table of Contents - Fixed Left Sidebar */}
      {tocItems.length > 0 && (
        <div
          className={`fixed left-4 z-40 hidden xl:block ${
            sidebarBottomOffset > 0
              ? ""
              : tocPosition === "center"
                ? "top-1/2 -translate-y-1/2"
                : "top-24 translate-y-0"
          }`}
          style={{
            width: tocCollapsed ? "48px" : "256px",
            transition: "width 400ms cubic-bezier(0.4, 0, 0.2, 1)",
            ...(sidebarBottomOffset > 0
              ? {
                  top: "auto",
                  bottom: `${sidebarBottomOffset}px`,
                }
              : {}),
          }}
        >
          <div
            className={`rounded-xl shadow-lg border backdrop-blur-md max-h-[65vh] flex flex-col ${
              showToc
                ? "opacity-100 translate-x-0"
                : "opacity-0 -translate-x-full"
            } ${tocCollapsed ? "overflow-hidden p-2" : "overflow-hidden"}`}
            style={{
              backgroundColor: "var(--background)/95",
              borderColor: "var(--border)",
              transition: "all 400ms cubic-bezier(0.4, 0, 0.2, 1)",
            }}
          >
            {/* Collapsed State - Icon Button */}
            {tocCollapsed && (
              <div
                className="flex flex-col items-center justify-center"
                style={{
                  opacity: tocCollapsed ? 1 : 0,
                  transition: "opacity 300ms cubic-bezier(0.4, 0, 0.2, 1)",
                }}
              >
                <button
                  onClick={() => setTocCollapsed(false)}
                  className="w-8 h-8 rounded-lg flex items-center justify-center sidebar-btn-hover"
                  style={{
                    color: "var(--text-secondary)",
                    backgroundColor: "var(--surface)",
                  }}
                  aria-label="Expand table of contents"
                  title="Table of Contents"
                >
                  {/* TOC icon - list with indentation */}
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
                      d="M4 6h16M4 12h12M4 18h8"
                    />
                  </svg>
                </button>
              </div>
            )}

            {/* Expanded State - Full TOC */}
            {!tocCollapsed && (
              <>
                {/* Fixed Header */}
                <div
                  className="flex items-center justify-between p-4 pb-3 flex-shrink-0 border-b"
                  style={{
                    borderColor: "var(--border)",
                  }}
                >
                  <h3
                    className="text-sm font-semibold whitespace-nowrap"
                    style={{
                      color: "var(--text-primary)",
                    }}
                  >
                    Table of Contents
                  </h3>

                  <button
                    onClick={() => setTocCollapsed(true)}
                    className="p-1.5 rounded-lg sidebar-btn-hover"
                    style={{ color: "var(--text-secondary)" }}
                    aria-label="Collapse table of contents"
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
                        d="M11 19l-7-7 7-7M19 19l-7-7 7-7"
                      />
                    </svg>
                  </button>
                </div>

                {/* Scrollable TOC List */}
                <nav className="space-y-1 p-4 pt-3 overflow-y-auto flex-1 toc-scrollbar">
                  {tocItems.map((item) => (
                    <button
                      key={item.id}
                      data-toc-id={item.id}
                      onClick={() => scrollToSection(item.id)}
                      className={`block w-full text-left py-2 px-3 rounded text-sm toc-btn-hover ${
                        activeSection === item.id
                          ? "font-medium"
                          : "font-normal"
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
                      title={item.title}
                    >
                      <span
                        className="line-clamp-2 leading-tight toc-item-title"
                        dangerouslySetInnerHTML={{ __html: item.titleHtml }}
                      />
                    </button>
                  ))}
                </nav>
              </>
            )}
          </div>
        </div>
      )}

      {/* TOC Toggle Button - Mobile & Tablet - Top Left */}
      {tocItems.length > 0 && (
        <button
          onClick={() => setShowToc(!showToc)}
          className="fixed left-4 top-20 z-40 xl:hidden w-10 h-10 rounded-full shadow-lg border transition-all duration-300 flex items-center justify-center"
          style={{
            backgroundColor: showToc ? "var(--accent)" : "var(--background)",
            borderColor: "var(--border)",
            color: showToc ? "white" : "var(--text-secondary)",
          }}
          aria-label="Toggle table of contents"
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
              d="M4 6h16M4 12h12M4 18h8"
            />
          </svg>
        </button>
      )}

      {/* Mobile TOC Bottom Sheet */}
      {tocItems.length > 0 && showToc && (
        <div
          className="fixed inset-0 z-50 xl:hidden"
          onClick={(e) => {
            if (e.target === e.currentTarget) setShowToc(false);
          }}
        >
          {/* Backdrop */}
          <div
            className="absolute inset-0"
            style={{ backgroundColor: "rgba(0, 0, 0, 0.5)" }}
          />

          {/* Bottom Sheet */}
          <div
            className="absolute bottom-0 left-0 right-0 rounded-t-2xl shadow-2xl max-h-[70vh] overflow-hidden flex flex-col"
            style={{
              backgroundColor: "var(--background)",
              animation: "slideUp 0.3s ease-out",
            }}
          >
            {/* Handle bar */}
            <div className="flex justify-center pt-3 pb-2">
              <div
                className="w-10 h-1 rounded-full"
                style={{ backgroundColor: "var(--border)" }}
              />
            </div>

            {/* Header */}
            <div
              className="flex items-center justify-between px-4 pb-3 border-b"
              style={{ borderColor: "var(--border)" }}
            >
              <h3
                className="text-base font-semibold"
                style={{ color: "var(--text-primary)" }}
              >
                Table of Contents
              </h3>
              <button
                onClick={() => setShowToc(false)}
                className="p-2 rounded-lg transition-colors duration-200"
                style={{ color: "var(--text-secondary)" }}
                aria-label="Close table of contents"
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
                    strokeWidth="2"
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>

            {/* TOC List */}
            <nav className="flex-1 overflow-y-auto p-4 space-y-1 toc-scrollbar">
              {tocItems.map((item) => (
                <button
                  key={item.id}
                  data-toc-id={item.id}
                  onClick={() => {
                    scrollToSection(item.id);
                    setShowToc(false);
                  }}
                  className={`block w-full text-left py-3 px-4 rounded-lg text-sm transition-all duration-200 ${
                    activeSection === item.id ? "font-medium" : "font-normal"
                  }`}
                  style={{
                    paddingLeft: `${16 + (item.level - 1) * 12}px`,
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
                        ? `3px solid var(--accent)`
                        : `3px solid transparent`,
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

            {/* Safe area padding for iOS */}
            <div
              className="h-6"
              style={{ backgroundColor: "var(--background)" }}
            />
          </div>

        </div>
      )}

      {/* Blog Graph Sidebar - Combined Reading Options + Related Articles Network */}
      <BlogGraphSidebar
        currentSlug={postSlug}
        tocPosition={tocPosition}
        sidebarBottomOffset={sidebarBottomOffset}
        isReadingMode={isReadingMode}
        onReadingModeToggle={toggleReadingMode}
        // Audio reading props
        isPlaying={isPlaying}
        isPaused={isPaused}
        progress={progress}
        duration={duration}
        remainingTime={remainingTime}
        onStartSpeech={startSpeech}
        onPauseSpeech={pauseSpeech}
        onResumeSpeech={resumeSpeech}
        onStopSpeech={stopSpeech}
        onSeekSpeech={seekSpeech}
        // Theme
        theme={theme}
      />

      {/* Main Content - Always Centered */}
      <div className="relative z-10 w-full overflow-x-hidden">
        <div className="flex justify-center items-start py-12 px-4 sm:px-6">
          <div
            className="w-full max-w-4xl mb-20"
            ref={contentRef}
            style={{
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
                className="text-2xl md:text-3xl lg:text-4xl font-bold mb-8 leading-tight"
                style={{
                  color: readingColors.textPrimary,
                  lineHeight: 1.3,
                }}
              >
                {title}
              </h1>

              {/* Article Meta */}
              {(author || publishDate || readTime || category) && (
              <div className="flex flex-wrap items-center justify-center gap-4 mb-6">
                {author && (
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
                          ? "#e8d5b7"
                          : "#92400e"
                        : readingColors.textPrimary,
                    }}
                  >
                    {author}
                  </span>
                </div>
                )}
                {publishDate && (
                <div className="flex items-center gap-1">
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    style={{
                      color: readingColors.textSecondary,
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
                      color: readingColors.textSecondary,
                    }}
                  >
                    {formatDate(publishDate)}
                  </time>
                </div>
                )}
                {readTime && (
                <div className="flex items-center gap-1">
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    style={{
                      color: readingColors.textSecondary,
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
                      color: readingColors.textSecondary,
                    }}
                  >
                    {readTime}
                  </span>
                </div>
                )}
                {category && (
                <span
                  className="px-3 py-1 rounded-full text-xs font-medium"
                  style={{
                    backgroundColor: readingColors.surfaceAccent,
                    color: readingColors.accent,
                  }}
                >
                  {category}
                </span>
                )}
              </div>
              )}

              {/* Tags */}
              {tags.length > 0 && (
                <div className="flex flex-wrap justify-center gap-2 mb-6">
                  <TagList tags={tags} variant="default" clickable={true} />
                </div>
              )}

              {aiGenerated && (
                <div className="flex justify-center mb-6">
                  <AiGeneratedBadge variant="detailed" />
                </div>
              )}
            </header>

            {/* Article Content */}
            <article
              ref={articleRef}
              className="prose prose-xl max-w-none mx-auto mb-16"
              style={{ color: readingColors.textPrimary }}
            >
              <MathJax className="blog-content" style={readingCssVars}>
                {dangerouslySetInnerHTML ? (
                  <div dangerouslySetInnerHTML={dangerouslySetInnerHTML} />
                ) : (
                  children
                )}
              </MathJax>
            </article>
          </div>
        </div>
      </div>

      {postSlug && (
        <BlogHighlighter slug={postSlug} containerRef={articleRef} />
      )}

      {/* Image Lightbox Modal */}
      {lightboxImage && (
        <div
          className="fixed inset-0 z-[100] flex items-center justify-center"
          onClick={() => setLightboxImage(null)}
          style={{
            backgroundColor: "rgba(0, 0, 0, 0.9)",
            animation: "fadeIn 0.2s ease-out",
          }}
        >
          {/* Close button */}
          <button
            onClick={() => setLightboxImage(null)}
            className="absolute top-4 right-4 p-2 rounded-full transition-all duration-200 hover:scale-110"
            style={{
              backgroundColor: "rgba(255, 255, 255, 0.1)",
              color: "white",
            }}
            aria-label="Close lightbox"
          >
            <svg
              className="w-6 h-6"
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

          {/* Image container */}
          <div
            className="relative max-w-[90vw] max-h-[90vh] flex items-center justify-center"
            onClick={(e) => e.stopPropagation()}
          >
            <img
              src={lightboxImage}
              alt={lightboxAlt}
              className="max-w-full max-h-[90vh] object-contain rounded-lg shadow-2xl"
              style={{
                animation: "zoomIn 0.2s ease-out",
              }}
            />
            {lightboxAlt && (
              <p
                className="absolute -bottom-10 left-0 right-0 text-center text-sm"
                style={{ color: "rgba(255, 255, 255, 0.7)" }}
              >
                {lightboxAlt}
              </p>
            )}
          </div>

        </div>
      )}
    </div>
  );
}
