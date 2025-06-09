"use client";

import React, {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
} from "react";
import { usePathname } from "next/navigation";

type ThemeContextType = {
  theme: "light" | "dark";
  toggleTheme: () => void;
  mounted: boolean;
  isReadingMode: boolean;
  setReadingMode: (mode: boolean) => void;
};

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [mounted, setMounted] = useState(false);
  const [isReadingMode, setIsReadingMode] = useState(false);
  const pathname = usePathname();

  // Function to check if current route should support reading mode
  const isBlogRoute = useCallback((path: string) => {
    return path.startsWith("/blog") || path.includes("/articles");
  }, []);

  // Initialize theme on mount
  useEffect(() => {
    const initTheme = () => {
      try {
        // Get saved theme or system preference
        const savedTheme = localStorage.getItem("theme");
        if (savedTheme === "light" || savedTheme === "dark") {
          return savedTheme;
        }

        // Check system preference
        return window.matchMedia("(prefers-color-scheme: dark)").matches
          ? "dark"
          : "light";
      } catch {
        return "light";
      }
    };

    const initialTheme = initTheme();
    setTheme(initialTheme);
    setMounted(true);
  }, []);

  // Automatically manage reading mode based on route changes
  useEffect(() => {
    if (!mounted) return;

    if (isBlogRoute(pathname)) {
      // Entering a blog route - restore reading mode if it was previously enabled
      try {
        const savedReadingMode = localStorage.getItem("readingMode");
        if (savedReadingMode === "true" && !isReadingMode) {
          setIsReadingMode(true);
        }
      } catch {
        // Ignore if localStorage is not available
      }
    } else if (isReadingMode) {
      // Leaving blog routes - turn off reading mode
      setIsReadingMode(false);
      try {
        localStorage.setItem("readingMode", "false");
      } catch {
        // Ignore if localStorage is not available
      }
    }
  }, [pathname, isReadingMode, isBlogRoute, mounted]);

  // Apply theme changes to DOM
  useEffect(() => {
    if (!mounted) return;

    const root = document.documentElement;

    // Clear any existing theme classes first to avoid conflicts
    root.classList.remove("dark", "reading-mode");

    // Only toggle 'dark' class
    if (theme === "dark") {
      root.classList.add("dark");
    }

    // Apply reading mode class
    if (isReadingMode) {
      root.classList.add("reading-mode");
    }

    // Set color scheme for native elements
    root.style.colorScheme = theme;

    // Save to localStorage
    try {
      localStorage.setItem("theme", theme);
    } catch {
      // Ignore if localStorage is not available
    }
  }, [theme, mounted, isReadingMode]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === "light" ? "dark" : "light"));
  };

  const setReadingMode = useCallback((mode: boolean) => {
    setIsReadingMode(mode);
    try {
      localStorage.setItem("readingMode", mode.toString());
    } catch {
      // Ignore if localStorage is not available
    }
  }, []);

  return (
    <ThemeContext.Provider
      value={{
        theme,
        toggleTheme,
        mounted,
        isReadingMode,
        setReadingMode,
      }}
    >
      {children}
    </ThemeContext.Provider>
  );
}

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return context;
};
