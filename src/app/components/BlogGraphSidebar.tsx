"use client";

import React, { useState } from "react";
import BlogGraphView from "./BlogGraphView";

interface BlogGraphSidebarProps {
  currentSlug?: string;
  tocPosition?: "center" | "top";
  // Reading options props
  isReadingMode?: boolean;
  onReadingModeToggle?: () => void;
  fontSize?: number;
  onFontSizeChange?: (size: number) => void;
  lineHeight?: number;
  onLineHeightChange?: (height: number) => void;
  // Audio reading props
  isPlaying?: boolean;
  isPaused?: boolean;
  progress?: number;
  duration?: number;
  remainingTime?: number;
  onStartSpeech?: () => void;
  onPauseSpeech?: () => void;
  onResumeSpeech?: () => void;
  onStopSpeech?: () => void;
  onSeekSpeech?: (percentage: number) => void;
  // Theme
  theme?: string;
}

export default function BlogGraphSidebar({
  currentSlug,
  tocPosition = "center",
  isReadingMode = false,
  onReadingModeToggle,
  fontSize = 18,
  onFontSizeChange,
  lineHeight = 1.6,
  onLineHeightChange,
  // Audio reading
  isPlaying = false,
  isPaused = false,
  progress = 0,
  duration = 0,
  remainingTime = 0,
  onStartSpeech,
  onPauseSpeech,
  onResumeSpeech,
  onStopSpeech,
  onSeekSpeech,
  // Theme
  theme = "light",
}: BlogGraphSidebarProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(false);

  const handleFontSizeChange = (newSize: number) => {
    if (onFontSizeChange) {
      onFontSizeChange(Math.max(12, Math.min(24, newSize)));
    }
  };

  const handleLineHeightChange = (newHeight: number) => {
    if (onLineHeightChange) {
      onLineHeightChange(Math.max(1.2, Math.min(2.0, newHeight)));
    }
  };

  // Graph background colors based on theme
  const graphBgColor = theme === "dark" ? "#1a1a2e" : "#f8fafc";
  const graphBgColorExpanded = theme === "dark" ? "#1a1a2e" : "#f1f5f9";

  return (
    <>
      {/* Sidebar Widget - Combined Reading Options + Graph */}
      <div
        className={`fixed right-4 z-40 hidden xl:block ${
          tocPosition === "center"
            ? "top-1/2 -translate-y-1/2"
            : "top-24 translate-y-0"
        }`}
        style={{
          width: isCollapsed ? "48px" : "280px",
          maxHeight: "calc(100vh - 120px)",
          transition:
            "width 400ms cubic-bezier(0.4, 0, 0.2, 1), top 500ms cubic-bezier(0.4, 0, 0.2, 1), transform 500ms cubic-bezier(0.4, 0, 0.2, 1)",
        }}
      >
        <div
          className={`rounded-xl shadow-lg border backdrop-blur-md flex flex-col overflow-hidden ${
            isCollapsed ? "p-2" : ""
          }`}
          style={{
            backgroundColor: "var(--background)",
            borderColor: "var(--border)",
            transition: "all 400ms cubic-bezier(0.4, 0, 0.2, 1)",
            maxHeight: "calc(100vh - 120px)",
          }}
        >
          {/* Collapsed State - Icon Buttons */}
          {isCollapsed && (
            <div className="flex flex-col items-center justify-center gap-2">
              <button
                onClick={() => setIsCollapsed(false)}
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
                aria-label="Expand sidebar"
                title="Reading Options & Graph"
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
                    d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4"
                  />
                </svg>
              </button>
            </div>
          )}

          {/* Expanded State - Full Widget */}
          {!isCollapsed && (
            <div className="flex flex-col overflow-y-auto overflow-x-hidden">
              {/* ============ READING OPTIONS SECTION ============ */}
              <div className="p-4 border-b" style={{ borderColor: "var(--border)" }}>
                <div className="flex items-center justify-between mb-3">
                  <h3
                    className="text-sm font-semibold"
                    style={{ color: "var(--text-primary)" }}
                  >
                    Reading Options
                  </h3>
                  <button
                    onClick={() => setIsCollapsed(true)}
                    className="p-1.5 rounded-lg transition-all duration-200"
                    style={{ color: "var(--text-secondary)" }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor = "var(--surface-accent)";
                      e.currentTarget.style.color = "var(--accent)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = "transparent";
                      e.currentTarget.style.color = "var(--text-secondary)";
                    }}
                    aria-label="Collapse sidebar"
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
                        d="M13 5l7 7-7 7M5 5l7 7-7 7"
                      />
                    </svg>
                  </button>
                </div>

                {/* Reading Mode Toggle */}
                <div className="flex items-center justify-between mb-3">
                  <span
                    className="text-xs"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Eye Comfort
                  </span>
                  <button
                    onClick={onReadingModeToggle}
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
                        e.currentTarget.style.backgroundColor = "var(--surface-accent)";
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
                        e.currentTarget.style.backgroundColor = "var(--surface-accent)";
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
                <div className="mb-3">
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
                        e.currentTarget.style.backgroundColor = "var(--surface-accent)";
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
                        onChange={(e) => handleLineHeightChange(Number(e.target.value))}
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
                        e.currentTarget.style.backgroundColor = "var(--surface-accent)";
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

                {/* ============ AUDIO READING SECTION ============ */}
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
                      <button
                        onClick={onStartSpeech}
                        className="flex-1 px-3 py-2 rounded-lg transition-all duration-200 flex items-center justify-center gap-2"
                        style={{
                          backgroundColor: "var(--surface)",
                          color: "var(--text-primary)",
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.backgroundColor = "var(--surface-accent)";
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
                      <>
                        <button
                          onClick={isPaused ? onResumeSpeech : onPauseSpeech}
                          className="flex-1 px-3 py-2 rounded-lg transition-all duration-200 flex items-center justify-center gap-2"
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
                          <span className="text-xs font-medium">
                            {isPaused ? "Continue" : "Pause"}
                          </span>
                        </button>

                        <button
                          onClick={onStopSpeech}
                          className="px-3 py-2 rounded-lg transition-all duration-200 flex items-center justify-center gap-1"
                          style={{
                            backgroundColor: "var(--surface)",
                            color: "var(--text-secondary)",
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.backgroundColor = "var(--surface-accent)";
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
                    <div className="mb-2">
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
                          if (onSeekSpeech) {
                            onSeekSpeech(Math.max(0, Math.min(100, percentage)));
                          }
                        }}
                      >
                        <div
                          className="h-2 rounded-full transition-all duration-300"
                          style={{
                            backgroundColor: "var(--accent)",
                            width: `${progress}%`,
                          }}
                        />
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

              {/* ============ GRAPH SECTION ============ */}
              <div>
                {/* Graph Header */}
                <div
                  className="flex items-center justify-between p-3 border-b"
                  style={{ borderColor: "var(--border)" }}
                >
                  <div className="flex items-center gap-2">
                    <svg
                      className="w-4 h-4"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                      style={{ color: "var(--accent)" }}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"
                      />
                    </svg>
                    <h3
                      className="text-sm font-semibold"
                      style={{ color: "var(--text-primary)" }}
                    >
                      Related Articles
                    </h3>
                  </div>

                  {/* Expand to fullscreen button */}
                  <button
                    onClick={() => setIsExpanded(true)}
                    className="p-1.5 rounded-lg transition-all duration-200"
                    style={{ color: "var(--text-secondary)" }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor = "var(--surface-accent)";
                      e.currentTarget.style.color = "var(--accent)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = "transparent";
                      e.currentTarget.style.color = "var(--text-secondary)";
                    }}
                    aria-label="Expand graph"
                    title="Open fullscreen"
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
                        d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"
                      />
                    </svg>
                  </button>
                </div>

                {/* Graph Container */}
                <div
                  className="relative overflow-hidden"
                  style={{
                    height: "280px",
                    backgroundColor: graphBgColor,
                  }}
                >
                  <BlogGraphView
                    currentSlug={currentSlug}
                    isExpanded={false}
                    height={280}
                    theme={theme}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Fullscreen Modal */}
      {isExpanded && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center"
          style={{ backgroundColor: "rgba(0, 0, 0, 0.85)" }}
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              setIsExpanded(false);
            }
          }}
        >
          <div
            className="relative w-[90vw] h-[85vh] rounded-xl overflow-hidden shadow-2xl"
            style={{
              backgroundColor: graphBgColorExpanded,
              border: theme === "dark" ? "1px solid rgba(255,255,255,0.1)" : "1px solid rgba(0,0,0,0.1)",
            }}
          >
            <BlogGraphView
              currentSlug={currentSlug}
              isExpanded={true}
              onClose={() => setIsExpanded(false)}
              theme={theme}
            />
          </div>
        </div>
      )}
    </>
  );
}
