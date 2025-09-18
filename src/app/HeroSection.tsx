"use client";

import Link from "next/link";
import Image from "next/image";
import React, { useState, useEffect } from "react";
import { useTheme } from "./ThemeProvider";

const titles = [
  "a Developer",
  "an AI Engineer",
  "a Problem Solver",
  "a Tech Enthusiast",
];

type Particle = {
  id: string;
  size: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  opacity: number;
  animationDelay: string;
};

// Helper function to generate deterministic particles for the background
// Uses an index-based approach instead of random to prevent hydration errors
const generateParticles = (count: number): Particle[] => {
  return Array.from({ length: count }, (_, index) => {
    // Create deterministic values based on index
    return {
      id: `particle-${index}`,
      size: 1 + (index % 3) + (index % 7) / 10,
      x: (index * 7.3) % 100,
      y: (index * 11.9) % 100,
      vx: ((index % 5) - 2) * 0.1,
      vy: ((index % 7) - 3) * 0.1,
      opacity: 0.2 + (index % 10) / 10,
      animationDelay: `${(index % 10) * 0.5}s`,
    };
  });
};

export default function HeroSection() {
  const [currentTitle, setCurrentTitle] = useState(0);
  const [displayText, setDisplayText] = useState("");
  const [isDeleting, setIsDeleting] = useState(false);
  const { theme, mounted } = useTheme();
  const [particles] = useState(() => generateParticles(80)); // Floating particles

  // State for dynamic statistics
  const [projectStats, setProjectStats] = useState<{
    total: number;
    totalStars: number;
  } | null>(null);

  // Get the appropriate profile image based on theme
  const getProfileImage = () => {
    if (!mounted) {
      // Return light mode image as fallback during hydration
      return "/profile-light-mode.webp";
    }
    return theme === "dark"
      ? "/profile-dark-mode.webp"
      : "/profile-light-mode.webp";
  };

  // Get theme-specific colors for the outer glow
  const getGlowColor = () => {
    return theme === "dark"
      ? "rgba(130, 170, 255, 0.7)" // Enhanced brighter blue glow for dark mode
      : "rgba(14, 165, 233, 0.6)"; // Enhanced sky blue glow for light mode
  };

  // Get secondary glow color for additional depth
  const getSecondaryGlowColor = () => {
    return theme === "dark"
      ? "rgba(79, 70, 229, 0.5)" // Enhanced indigo for dark mode
      : "rgba(56, 189, 248, 0.4)"; // Enhanced lighter blue for light mode
  };

  // Get tertiary glow color for more dynamic effect
  const getTertiaryGlowColor = () => {
    return theme === "dark"
      ? "rgba(192, 132, 252, 0.2)" // Purple hint for dark mode
      : "rgba(2, 132, 199, 0.25)"; // Deeper blue for light mode
  };

  useEffect(() => {
    const currentString = titles[currentTitle];
    const timeout = setTimeout(
      () => {
        if (!isDeleting) {
          if (displayText.length < currentString.length) {
            setDisplayText(currentString.slice(0, displayText.length + 1));
          } else {
            setTimeout(() => setIsDeleting(true), 2000);
          }
        } else {
          if (displayText.length > 0) {
            setDisplayText(displayText.slice(0, -1));
          } else {
            setIsDeleting(false);
            setCurrentTitle((prev) => (prev + 1) % titles.length);
          }
        }
      },
      isDeleting ? 50 : 100
    );

    return () => clearTimeout(timeout);
  }, [displayText, isDeleting, currentTitle]);

  // Fetch project statistics
  useEffect(() => {
    const fetchProjectStats = async () => {
      try {
        const response = await fetch("/api/projects");
        if (response.ok) {
          const data = await response.json();
          const stats = {
            total: data.total || 0,
            totalStars:
              data.projects?.reduce(
                (sum: number, project: { stars?: number }) =>
                  sum + (project.stars || 0),
                0
              ) || 0,
          };
          setProjectStats(stats);
        }
      } catch (error) {
        console.error("Failed to fetch project stats:", error);
        // Use fallback values if API fails
        setProjectStats({ total: 8, totalStars: 1200 });
      }
    };

    fetchProjectStats();
  }, []);

  return (
    <section
      className="relative py-20 md:py-28 px-6 min-h-[calc(100vh-80px)] overflow-hidden section-primary"
      aria-label="Hero section with introduction and main call-to-action"
      role="banner"
    >
      {/* Enhanced Particle Background */}
      <div
        className="absolute inset-0 -z-30 overflow-hidden"
        aria-hidden="true"
        style={{
          backgroundColor:
            theme === "dark"
              ? "rgba(15, 23, 42, 0.6)"
              : "rgba(248, 250, 252, 0.8)",
        }}
      >
        {/* Main floating particles */}
        {particles.map((particle) => (
          <div
            key={particle.id}
            className="absolute rounded-full animate-float-slow"
            style={{
              width: `${particle.size}px`,
              height: `${particle.size}px`,
              left: `${particle.x}%`,
              top: `${particle.y}%`,
              backgroundColor:
                theme === "dark"
                  ? `rgba(148, 163, 184, ${particle.opacity * 0.6})`
                  : `rgba(71, 85, 105, ${particle.opacity * 0.4})`,
              boxShadow:
                theme === "dark"
                  ? `0 0 ${particle.size * 1.5}px rgba(148, 163, 184, ${
                      particle.opacity * 0.3
                    })`
                  : `0 0 ${particle.size * 1.5}px rgba(71, 85, 105, ${
                      particle.opacity * 0.2
                    })`,
              animationDelay: particle.animationDelay,
              animationDuration: `${4 + (particle.id.charCodeAt(9) % 6)}s`,
            }}
          ></div>
        ))}

        {/* Accent particles for visual interest */}
        {particles.slice(0, 30).map((particle) => (
          <div
            key={`accent-${particle.id}`}
            className="absolute rounded-full animate-pulse-slow"
            style={{
              width: `${particle.size * 0.7}px`,
              height: `${particle.size * 0.7}px`,
              left: `${(particle.x + 15) % 100}%`,
              top: `${(particle.y + 25) % 100}%`,
              backgroundColor:
                theme === "dark"
                  ? `rgba(59, 130, 246, ${particle.opacity * 0.4})`
                  : `rgba(37, 99, 235, ${particle.opacity * 0.3})`,
              boxShadow:
                theme === "dark"
                  ? `0 0 ${particle.size}px rgba(59, 130, 246, ${
                      particle.opacity * 0.2
                    })`
                  : `0 0 ${particle.size}px rgba(37, 99, 235, ${
                      particle.opacity * 0.15
                    })`,
              animationDelay: `${(particle.id.charCodeAt(9) % 10) * 0.5}s`,
              animationDuration: `${8 + (particle.id.charCodeAt(9) % 6)}s`,
            }}
          ></div>
        ))}
      </div>

      {/* Grid Background Pattern */}
      <div className="absolute inset-0 -z-20">
        <div
          className="absolute inset-0 bg-[size:24px_24px] opacity-30"
          style={{
            backgroundImage: `linear-gradient(to right, var(--border) 1px, transparent 1px), linear-gradient(to bottom, var(--border) 1px, transparent 1px)`,
          }}
        ></div>
      </div>

      {/* Background Effects */}
      <div className="absolute inset-0 -z-10 overflow-hidden">
        <div
          className="absolute top-32 left-16 w-72 h-72 blur-3xl rounded-full animate-float-slow"
          style={{
            background:
              "linear-gradient(135deg, var(--accent) 20%, var(--accent-light) 40%)",
            opacity: theme === "dark" ? 0.15 : 0.1,
          }}
        ></div>
        <div
          className="absolute bottom-32 right-16 w-96 h-96 blur-3xl rounded-full animate-float-slow"
          style={{
            background:
              "linear-gradient(135deg, var(--accent-light) 20%, var(--accent) 40%)",
            opacity: theme === "dark" ? 0.15 : 0.1,
            animationDelay: "1s",
          }}
        ></div>
      </div>

      <div className="flex flex-col lg:flex-row items-center justify-between gap-12 lg:gap-16 h-full max-w-6xl mx-auto">
        {/* Left Content */}
        <header className="flex-1 max-w-2xl">
          <div className="mb-6">
            <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold mb-6 leading-tight">
              <span
                className="font-normal text-lg sm:text-xl md:text-2xl block mb-2"
                style={{ color: "var(--text-secondary)" }}
              >
                Hi! I am
              </span>
              <div className="inline-flex items-baseline">
                <span className="sr-only">
                  AI Engineer and Full-Stack Developer
                </span>
              </div>
              <div className="inline-flex items-baseline mt-2">
                <span className="gradient-text" aria-live="polite">
                  {displayText}
                </span>
                <span
                  className="animate-blink border-r-2 ml-1 inline-block"
                  style={{
                    borderColor: "var(--accent)",
                    height: "1em",
                    width: "2px",
                  }}
                  aria-hidden="true"
                ></span>
              </div>
            </h1>
          </div>

          <h2
            className="text-xl sm:text-2xl md:text-3xl font-semibold mb-6 leading-relaxed"
            style={{ color: "var(--text-primary)" }}
          >
            Building the future, one line at a time
          </h2>

          <p
            className="text-base md:text-lg mb-8 leading-relaxed"
            style={{ color: "var(--text-secondary)" }}
          >
            Welcome to my corner of the internet! I&apos;m a passionate AI
            Engineer and Full-Stack Developer sharing insights on Machine
            Learning, Deep Learning, Software Engineering, and the journey of
            building meaningful technology that makes a difference.
          </p>

          <nav
            className="flex flex-col sm:flex-row gap-4 mb-10"
            aria-label="Main navigation buttons"
          >
            <Link
              href="/blog"
              className="hero-button-primary inline-flex items-center justify-center px-6 py-3 font-medium rounded-lg transition-all duration-300 shadow-sm text-white gap-2 relative overflow-hidden group"
              style={{
                backgroundColor: "var(--accent)",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = "var(--accent-hover)";
                e.currentTarget.style.transform = "translateY(-2px)";
                e.currentTarget.style.boxShadow =
                  "0 10px 25px -5px var(--accent)/40, 0 8px 10px -6px var(--accent)/20";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = "var(--accent)";
                e.currentTarget.style.transform = "translateY(0)";
                e.currentTarget.style.boxShadow =
                  "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)";
              }}
              aria-describedby="blog-description"
            >
              {/* Ripple effect */}
              <span className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 transform -skew-x-12 group-hover:animate-shine"></span>
              <svg
                className="w-5 h-5 relative z-10"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"
                />
              </svg>
              <span className="relative z-10">Read My Blog</span>
            </Link>
            <Link
              href="/projects"
              className="hero-button-secondary inline-flex items-center justify-center px-6 py-3 font-medium rounded-lg transition-all duration-300 border gap-2 relative overflow-hidden group"
              style={{
                backgroundColor: "var(--card-bg)",
                borderColor: "var(--border)",
                color: "var(--text-primary)",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = "var(--surface)";
                e.currentTarget.style.borderColor = "var(--accent)";
                e.currentTarget.style.color = "var(--accent)";
                e.currentTarget.style.transform = "translateY(-2px)";
                e.currentTarget.style.boxShadow =
                  "0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = "var(--card-bg)";
                e.currentTarget.style.borderColor = "var(--border)";
                e.currentTarget.style.color = "var(--text-primary)";
                e.currentTarget.style.transform = "translateY(0)";
                e.currentTarget.style.boxShadow =
                  "0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)";
              }}
              aria-describedby="projects-description"
            >
              {/* Subtle hover effect */}
              <span className="absolute inset-0 bg-gradient-to-r from-transparent via-[var(--accent)]/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></span>
              <svg
                className="w-5 h-5 relative z-10"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
                />
              </svg>
              <span className="relative z-10">View Projects</span>
            </Link>
            <div className="sr-only">
              <p id="blog-description">
                Explore my latest blog posts about AI, machine learning, and
                software development
              </p>
              <p id="projects-description">
                View my portfolio of AI and web development projects
              </p>
            </div>
          </nav>

          {/* Connect Section - Simplified with smaller icons */}
          <div className="mb-10">
            <p
              className="mb-3 font-medium text-sm"
              style={{ color: "var(--text-muted)" }}
            >
              Connect with me
            </p>
            <div
              className="flex gap-4"
              role="list"
              aria-label="Social media links"
            >
              <Link
                href="https://github.com/hieptran1812"
                target="_blank"
                rel="noopener noreferrer"
                className="transition-colors duration-200"
                style={{ color: "var(--text-muted)" }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.color = "var(--accent)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.color = "var(--text-muted)";
                }}
                aria-label="Visit Hiep Tran's GitHub profile"
                role="listitem"
              >
                <svg
                  className="w-5 h-5"
                  fill="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path d="M12 0C5.374 0 0 5.373 0 12 0 17.302 3.438 21.8 8.207 23.387c.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0112 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z" />
                </svg>
              </Link>
              <Link
                href="https://www.linkedin.com/in/hieptran01"
                target="_blank"
                rel="noopener noreferrer"
                className="transition-colors duration-200"
                style={{ color: "var(--text-muted)" }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.color = "var(--accent)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.color = "var(--text-muted)";
                }}
                aria-label="Visit Hiep Tran's LinkedIn profile"
                role="listitem"
              >
                <svg
                  className="w-5 h-5"
                  fill="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
                </svg>
              </Link>
              <Link
                href="https://x.com/halleytran01"
                target="_blank"
                rel="noopener noreferrer"
                className="transition-colors duration-200 group"
                style={{ color: "var(--text-muted)" }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.color = "var(--accent)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.color = "var(--text-muted)";
                }}
                aria-label="Follow Hiep Tran on X"
                role="listitem"
              >
                <svg
                  className="w-5 h-5 transition-all duration-200 group-hover:scale-110"
                  viewBox="0 0 120 120"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                  style={{ display: "block" }}
                >
                  <rect width="120" height="120" rx="24" fill="#fff" />
                  <path
                    d="M86.4 33.6H99L72.6 62.1L103.2 99H80.7L61.8 76.2L40.8 99H28.2L56.1 68.1L27 33.6H50.1L66.3 53.7L86.4 33.6ZM82.2 92.1H88.8L48.3 40.2H41.1L82.2 92.1Z"
                    fill="#000"
                  />
                </svg>
              </Link>
              <Link
                href="mailto:hieptran.jobs@gmail.com"
                className="transition-colors duration-200"
                style={{ color: "var(--text-muted)" }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.color = "var(--accent)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.color = "var(--text-muted)";
                }}
                aria-label="Send email to Hiep Tran"
                role="listitem"
              >
                <svg
                  className="w-5 h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  strokeWidth="2"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M3 8l7.89 4.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                  />
                </svg>
              </Link>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div
              className="rounded-lg p-3 border shadow-sm"
              style={{
                backgroundColor: "var(--card-bg)",
                borderColor: "var(--card-border)",
              }}
            >
              <div
                className="text-2xl font-bold mb-1"
                style={{ color: "var(--text-primary)" }}
              >
                50+
              </div>
              <div className="text-xs" style={{ color: "var(--text-muted)" }}>
                Blog Posts
              </div>
            </div>
            <div
              className="rounded-lg p-3 border shadow-sm"
              style={{
                backgroundColor: "var(--card-bg)",
                borderColor: "var(--card-border)",
              }}
            >
              <div
                className="text-2xl font-bold mb-1"
                style={{ color: "var(--text-primary)" }}
              >
                {projectStats ? `${projectStats.total}+` : "5+"}
              </div>
              <div className="text-xs" style={{ color: "var(--text-muted)" }}>
                Projects
              </div>
            </div>
            <div
              className="rounded-lg p-3 border shadow-sm"
              style={{
                backgroundColor: "var(--card-bg)",
                borderColor: "var(--card-border)",
              }}
            >
              <div
                className="text-2xl font-bold mb-1"
                style={{ color: "var(--text-primary)" }}
              >
                {projectStats
                  ? projectStats.totalStars >= 1000
                    ? `${Math.floor(projectStats.totalStars / 1000)}K+`
                    : `${projectStats.totalStars}+`
                  : "4K+"}
              </div>
              <div className="text-xs" style={{ color: "var(--text-muted)" }}>
                GitHub Stars
              </div>
            </div>
            <div
              className="rounded-lg p-3 border shadow-sm"
              style={{
                backgroundColor: "var(--card-bg)",
                borderColor: "var(--card-border)",
              }}
            >
              <div
                className="text-2xl font-bold mb-1"
                style={{ color: "var(--text-primary)" }}
              >
                4+
              </div>
              <div className="text-xs" style={{ color: "var(--text-muted)" }}>
                Years Coding
              </div>
            </div>
          </div>
        </header>

        {/* Enhanced Right Image with Advanced Outer Glow Effect */}
        <aside
          className="flex-shrink-0 relative"
          aria-label="Profile image and availability status"
        >
          {/* Primary outer glow container - Fixed dimensions to prevent layout shift */}
          <div
            className="relative rounded-2xl"
            style={{
              width: "350px",
              height: "350px",
              maxWidth: "80vw",
              maxHeight: "80vw",
              minWidth: "280px",
              minHeight: "280px",
              boxShadow: `0 0 30px 12px ${getGlowColor()}, 0 0 60px 20px ${getSecondaryGlowColor()}, 0 0 90px 30px ${getTertiaryGlowColor()}`,
              transform: "translate3d(0, 0, 0)", // Hardware acceleration with better performance
              willChange: "auto", // Prevent unnecessary layer creation
            }}
          >
            {/* Actual image container with gradient border */}
            <div
              className="absolute inset-0 rounded-2xl p-[3px] overflow-hidden"
              style={{
                background:
                  theme === "dark"
                    ? "linear-gradient(135deg, rgba(130, 170, 255, 0.5) 0%, rgba(79, 70, 229, 0.5) 50%, rgba(130, 170, 255, 0.2) 100%)"
                    : "linear-gradient(135deg, rgba(14, 165, 233, 0.5) 0%, rgba(56, 189, 248, 0.5) 50%, rgba(14, 165, 233, 0.2) 100%)",
                boxShadow: "inset 0 0 10px rgba(0,0,0,0.1)",
              }}
            >
              <div className="w-full h-full rounded-xl overflow-hidden">
                <Image
                  src={getProfileImage()}
                  alt="Hiep Tran - AI Engineer and Full-Stack Developer profile photo"
                  width={350}
                  height={350}
                  className="rounded-xl shadow-lg object-cover"
                  style={{
                    width: "100%",
                    height: "100%",
                    objectFit: "cover",
                  }}
                  priority
                  sizes="(max-width: 768px) 280px, 350px"
                  placeholder="blur"
                  blurDataURL="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAAIAAoDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAhEAACAQMDBQAAAAAAAAAAAAABAgMABAUGIWGRkbHB0f/EABUBAQEAAAAAAAAAAAAAAAAAAAMF/8QAGhEAAgIDAAAAAAAAAAAAAAAAAAECEgMRkf/aAAwDAQACEQMRAD8AltJagyeH0AthI5xdrLcNM91BF5pX2HaH9bcfaSXWGaRmknyJckliyjqTzSlT54b6bk+h0R//2Q=="
                />
              </div>
            </div>

            {/* Available Badge - More subtle and clean */}
            <div
              className="absolute -bottom-2 -right-2 rounded-full px-3 py-1.5 shadow-md border flex items-center gap-2"
              style={{
                backgroundColor: "var(--card-bg)",
                borderColor: "var(--card-border)",
              }}
              role="status"
              aria-label="Availability status"
            >
              <div
                className="w-2 h-2 rounded-full animate-pulse"
                style={{ backgroundColor: "var(--accent-light)" }}
                aria-hidden="true"
              ></div>
              <span
                className="text-xs font-medium"
                style={{ color: "var(--text-primary)" }}
              >
                Available for collaboration
              </span>
            </div>
          </div>
        </aside>
      </div>
    </section>
  );
}
