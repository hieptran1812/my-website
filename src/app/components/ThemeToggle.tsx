"use client";

import React from "react";
import { useTheme } from "../ThemeProvider";

export default function ThemeToggle() {
  const { theme, toggleTheme, mounted } = useTheme();

  // Show a placeholder during SSR/hydration
  if (!mounted) {
    return (
      <div className="relative inline-flex h-8 w-14 items-center justify-center rounded-full border border-gray-200 bg-white/95 backdrop-blur-md transition-all duration-300 dark:border-gray-700 dark:bg-gray-800/95">
        <div className="absolute left-1 top-1 h-6 w-6 rounded-full bg-gradient-to-r from-yellow-400 to-orange-500 shadow-lg" />
        <span className="sr-only">Loading theme toggle</span>
      </div>
    );
  }

  const handleToggle = () => {
    console.log("Theme toggle clicked, current theme:", theme);
    console.log(
      "HTML classes before toggle:",
      document.documentElement.classList.toString()
    );
    toggleTheme();
  };

  return (
    <button
      onClick={handleToggle}
      className="relative inline-flex h-8 w-14 items-center justify-center rounded-full border border-gray-200 bg-white/95 backdrop-blur-md transition-all duration-300 hover:border-gray-300 hover:bg-white dark:border-gray-700 dark:bg-gray-800/95 dark:hover:border-gray-600 dark:hover:bg-gray-800"
      aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
    >
      <div
        className={`absolute top-1 h-6 w-6 rounded-full bg-gradient-to-r transition-all duration-300 shadow-lg ${
          theme === "light"
            ? "left-1 from-amber-400 to-orange-500"
            : "left-7 from-blue-500 to-blue-600"
        }`}
      />

      {/* Sun Icon */}
      <svg
        className={`absolute left-2 h-4 w-4 transition-all duration-300 ${
          theme === "light"
            ? "opacity-100 text-white"
            : "opacity-0 text-gray-400"
        }`}
        fill="currentColor"
        viewBox="0 0 24 24"
      >
        <path d="M12 2.25a.75.75 0 01.75.75v2.25a.75.75 0 01-1.5 0V3a.75.75 0 01.75-.75zM7.5 12a4.5 4.5 0 119 0 4.5 4.5 0 01-9 0zM18.894 6.166a.75.75 0 00-1.06-1.06l-1.591 1.59a.75.75 0 101.06 1.061l1.591-1.59zM21.75 12a.75.75 0 01-.75.75h-2.25a.75.75 0 010-1.5H21a.75.75 0 01.75.75zM17.834 18.894a.75.75 0 001.06-1.06l-1.59-1.591a.75.75 0 10-1.061 1.06l1.59 1.591zM12 18a.75.75 0 01.75.75V21a.75.75 0 01-1.5 0v-2.25A.75.75 0 0112 18zM7.758 17.303a.75.75 0 00-1.061-1.06l-1.591 1.59a.75.75 0 001.06 1.061l1.591-1.59zM6 12a.75.75 0 01-.75.75H3a.75.75 0 010-1.5h2.25A.75.75 0 016 12zM6.697 7.757a.75.75 0 001.06-1.06l-1.59-1.591a.75.75 0 00-1.061 1.06l1.59 1.591z" />
      </svg>

      {/* Moon Icon */}
      <svg
        className={`absolute right-2 h-4 w-4 transition-all duration-300 ${
          theme === "dark"
            ? "opacity-100 text-white"
            : "opacity-0 text-gray-400"
        }`}
        fill="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          fillRule="evenodd"
          d="M9.528 1.718a.75.75 0 01.162.819A8.97 8.97 0 009 6a9 9 0 009 9 8.97 8.97 0 003.463-.69.75.75 0 01.981.98 10.503 10.503 0 01-9.694 6.46c-5.799 0-10.5-4.701-10.5-10.5 0-4.368 2.667-8.112 6.46-9.694a.75.75 0 01.818.162z"
          clipRule="evenodd"
        />
      </svg>
    </button>
  );
}
