"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { useTheme } from "../ThemeProvider";
import CollectionTag from "../../components/CollectionTag";
import { TagList } from "../../components/TagBadge";
import "katex/dist/katex.min.css";
import MathJax from "./MathJax";
import "./BlogContent.css"; // Re-enabled for proper styling
import "../../components/styles/TextHighlight.css"; // Import text highlighting styles
import { SpeechReader } from "../../components/utils/SpeechReader";
import type {
  SpeechReaderOptions,
  SpeechReaderEvents,
} from "../../components/utils/SpeechReader";
import { formatDate } from "../../lib/dateUtils";
import BlogGraphSidebar from "./BlogGraphSidebar";

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
  const [isSmallScreen, setIsSmallScreen] = useState(false);

  // Image lightbox state
  const [lightboxImage, setLightboxImage] = useState<string | null>(null);
  const [lightboxAlt, setLightboxAlt] = useState<string>("");

  const contentRef = useRef<HTMLDivElement>(null);
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
  }, []);

  // Update active section based on scroll position - find heading closest to middle of screen
  useEffect(() => {
    if (tocItems.length === 0) return;

    const handleScroll = () => {
      const viewportHeight = window.innerHeight;
      const viewportMiddle = window.scrollY + viewportHeight / 2;

      // Find the heading closest to the middle of the viewport
      let closestId = tocItems[0]?.id || "";
      let closestDistance = Infinity;

      for (let i = 0; i < tocItems.length; i++) {
        const element = tocItems[i].element;
        if (element) {
          const rect = element.getBoundingClientRect();
          const elementMiddle = window.scrollY + rect.top + rect.height / 2;
          const distance = Math.abs(viewportMiddle - elementMiddle);

          if (distance < closestDistance) {
            closestDistance = distance;
            closestId = tocItems[i].id;
          }
        }
      }

      setActiveSection((prev) => {
        if (prev !== closestId) {
          // Auto-scroll TOC to show active item
          setTimeout(() => {
            const activeButton = document.querySelector(`[data-toc-id="${closestId}"]`);
            if (activeButton) {
              activeButton.scrollIntoView({
                behavior: "smooth",
                block: "nearest",
              });
            }
          }, 10);
        }
        return closestId;
      });
    };

    // Initial check
    handleScroll();

    window.addEventListener("scroll", handleScroll, { passive: true });

    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, [tocItems]);

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

  // Close lightbox on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && lightboxImage) {
        setLightboxImage(null);
      }
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
            ...(sidebarBottomOffset > 0 ? {
              top: "auto",
              bottom: `${sidebarBottomOffset}px`,
            } : {}),
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
                  className="w-8 h-8 rounded-lg flex items-center justify-center group"
                  style={{
                    color: "var(--text-secondary)",
                    backgroundColor: "var(--surface)",
                    transition: "all 300ms cubic-bezier(0.4, 0, 0.2, 1)",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = "var(--surface-accent)";
                    e.currentTarget.style.color = "var(--accent)";
                    e.currentTarget.style.transform = "scale(1.05)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = "var(--surface)";
                    e.currentTarget.style.color = "var(--text-secondary)";
                    e.currentTarget.style.transform = "scale(1)";
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
                    className="p-1.5 rounded-lg"
                    style={{
                      color: "var(--text-secondary)",
                      transition: "all 300ms cubic-bezier(0.4, 0, 0.2, 1)",
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor = "var(--surface-accent)";
                      e.currentTarget.style.color = "var(--accent)";
                      e.currentTarget.style.transform = "scale(1.1)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = "transparent";
                      e.currentTarget.style.color = "var(--text-secondary)";
                      e.currentTarget.style.transform = "scale(1)";
                    }}
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
            <div className="h-6" style={{ backgroundColor: "var(--background)" }} />
          </div>

          {/* Animation keyframes */}
          <style>{`
            @keyframes slideUp {
              from {
                transform: translateY(100%);
              }
              to {
                transform: translateY(0);
              }
            }
          `}</style>
        </div>
      )}

      {/* Blog Graph Sidebar - Combined Reading Options + Related Articles Network */}
      <BlogGraphSidebar
        currentSlug={postSlug}
        tocPosition={tocPosition}
        sidebarBottomOffset={sidebarBottomOffset}
        isReadingMode={isReadingMode}
        onReadingModeToggle={() => setReadingMode(!isReadingMode)}
        fontSize={fontSize}
        onFontSizeChange={handleFontSizeChange}
        lineHeight={lineHeight}
        onLineHeightChange={handleLineHeightChange}
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
                <div className="flex flex-wrap justify-center gap-2 mb-6">
                  <TagList
                    tags={tags}
                    variant="default"
                    clickable={true}
                  />
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

          </div>
        </div>
      </div>

      {/* Prevent duplicate TOC on small screens */}
      {tocItems.length > 0 && !isSmallScreen && showToc && (
        <div className="toc-container">{/* Render TOC */}</div>
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

          {/* Animation keyframes */}
          <style>{`
            @keyframes fadeIn {
              from { opacity: 0; }
              to { opacity: 1; }
            }
            @keyframes zoomIn {
              from {
                opacity: 0;
                transform: scale(0.9);
              }
              to {
                opacity: 1;
                transform: scale(1);
              }
            }
          `}</style>
        </div>
      )}
    </div>
  );
}
