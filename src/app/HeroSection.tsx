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

  return (
    <section
      className="relative py-12 md:py-20 lg:py-28 px-6 min-h-[calc(100vh-80px)] flex items-center overflow-hidden section-primary"
      aria-label="Hero section with introduction and main call-to-action"
      role="banner"
    >
      {/* Enhanced Particle Background */}
      <div
        className="absolute inset-0 -z-30 overflow-hidden"
        aria-hidden="true"
        style={{
          background:
            theme === "dark"
              ? "linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.85) 50%, rgba(15, 23, 42, 0.95) 100%)"
              : "linear-gradient(135deg, rgba(248, 250, 252, 0.95) 0%, rgba(241, 245, 249, 0.9) 50%, rgba(248, 250, 252, 0.95) 100%)",
        }}
      >
        {/* Optimized floating particles with better performance */}
        {particles.slice(0, 50).map((particle) => (
          <div
            key={particle.id}
            className="absolute rounded-full animate-float-slow"
            style={{
              width: `${particle.size * 0.8}px`,
              height: `${particle.size * 0.8}px`,
              left: `${particle.x}%`,
              top: `${particle.y}%`,
              backgroundColor:
                theme === "dark"
                  ? `rgba(99, 102, 241, ${particle.opacity * 0.4})`
                  : `rgba(59, 130, 246, ${particle.opacity * 0.3})`,
              boxShadow:
                theme === "dark"
                  ? `0 0 ${particle.size * 2}px rgba(99, 102, 241, ${
                      particle.opacity * 0.2
                    })`
                  : `0 0 ${particle.size * 1.5}px rgba(59, 130, 246, ${
                      particle.opacity * 0.15
                    })`,
              animationDelay: particle.animationDelay,
              animationDuration: `${6 + (particle.id.charCodeAt(9) % 8)}s`,
            }}
          ></div>
        ))}

        {/* Accent particles for depth */}
        {particles.slice(0, 20).map((particle) => (
          <div
            key={`accent-${particle.id}`}
            className="absolute rounded-full animate-pulse-slow"
            style={{
              width: `${particle.size * 0.5}px`,
              height: `${particle.size * 0.5}px`,
              left: `${(particle.x + 20) % 100}%`,
              top: `${(particle.y + 30) % 100}%`,
              backgroundColor:
                theme === "dark"
                  ? `rgba(168, 85, 247, ${particle.opacity * 0.3})`
                  : `rgba(147, 51, 234, ${particle.opacity * 0.25})`,
              boxShadow:
                theme === "dark"
                  ? `0 0 ${particle.size}px rgba(168, 85, 247, ${
                      particle.opacity * 0.15
                    })`
                  : `0 0 ${particle.size}px rgba(147, 51, 234, ${
                      particle.opacity * 0.1
                    })`,
              animationDelay: `${(particle.id.charCodeAt(9) % 15) * 0.3}s`,
              animationDuration: `${10 + (particle.id.charCodeAt(9) % 8)}s`,
            }}
          ></div>
        ))}
      </div>

      {/* Modern Grid Background Pattern */}
      <div className="absolute inset-0 -z-20">
        <div
          className="absolute inset-0 bg-[size:32px_32px] opacity-20"
          style={{
            backgroundImage: `
              linear-gradient(to right, ${
                theme === "dark"
                  ? "rgba(99, 102, 241, 0.1)"
                  : "rgba(59, 130, 246, 0.08)"
              } 1px, transparent 1px), 
              linear-gradient(to bottom, ${
                theme === "dark"
                  ? "rgba(99, 102, 241, 0.1)"
                  : "rgba(59, 130, 246, 0.08)"
              } 1px, transparent 1px)
            `,
          }}
        ></div>
      </div>

      {/* Enhanced Background Effects */}
      <div className="absolute inset-0 -z-10 overflow-hidden">
        <div
          className="absolute top-1/4 left-1/4 w-96 h-96 blur-3xl rounded-full animate-float-slow"
          style={{
            background:
              theme === "dark"
                ? "linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.1) 100%)"
                : "linear-gradient(135deg, rgba(59, 130, 246, 0.12) 0%, rgba(147, 51, 234, 0.08) 100%)",
            animationDuration: "8s",
          }}
        ></div>
        <div
          className="absolute bottom-1/4 right-1/4 w-80 h-80 blur-3xl rounded-full animate-float-slow"
          style={{
            background:
              theme === "dark"
                ? "linear-gradient(135deg, rgba(168, 85, 247, 0.12) 0%, rgba(59, 130, 246, 0.08) 100%)"
                : "linear-gradient(135deg, rgba(147, 51, 234, 0.1) 0%, rgba(59, 130, 246, 0.06) 100%)",
            animationDelay: "2s",
            animationDuration: "10s",
          }}
        ></div>
        <div
          className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-72 h-72 blur-3xl rounded-full animate-pulse-slow"
          style={{
            background:
              theme === "dark"
                ? "radial-gradient(circle, rgba(99, 102, 241, 0.08) 0%, transparent 70%)"
                : "radial-gradient(circle, rgba(59, 130, 246, 0.06) 0%, transparent 70%)",
            animationDuration: "12s",
          }}
        ></div>
      </div>

      <div className="container mx-auto max-w-7xl px-4">
        <div className="flex flex-col lg:flex-row items-center justify-center min-h-[calc(100vh-180px)] gap-6 lg:gap-0">
          {/* Enhanced Right Image with Modern Design */}
          <aside
            className="flex-1 flex justify-center items-center max-w-md lg:max-w-lg order-1 lg:order-1"
            aria-label="Profile image and availability status"
          >
            {/* Modern image container with enhanced glow */}
            <div
              className="relative group"
              style={{
                width: "320px",
                height: "320px",
                maxWidth: "80vw",
                maxHeight: "80vw",
                minWidth: "280px",
                minHeight: "280px",
                transform: "translateY(-35px)",
              }}
            >
              {/* Outer glow effects with improved performance */}
              <div
                className="absolute inset-0 rounded-3xl blur-xl transition-all duration-700 group-hover:blur-2xl"
                style={{
                  background:
                    theme === "dark"
                      ? "linear-gradient(135deg, rgba(99, 102, 241, 0.4) 0%, rgba(168, 85, 247, 0.3) 50%, rgba(59, 130, 246, 0.2) 100%)"
                      : "linear-gradient(135deg, rgba(59, 130, 246, 0.3) 0%, rgba(147, 51, 234, 0.25) 50%, rgba(99, 102, 241, 0.15) 100%)",
                  transform: "scale(1.1)",
                  animation: "pulse 4s ease-in-out infinite",
                }}
              ></div>

              {/* Secondary glow for depth */}
              <div
                className="absolute inset-0 rounded-3xl blur-2xl opacity-60 transition-all duration-700 group-hover:opacity-80"
                style={{
                  background:
                    theme === "dark"
                      ? "radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, transparent 70%)"
                      : "radial-gradient(circle, rgba(59, 130, 246, 0.2) 0%, transparent 70%)",
                  transform: "scale(1.2)",
                  animation: "pulse 6s ease-in-out infinite reverse",
                }}
              ></div>

              {/* Image container with gradient border */}
              <div
                className="relative rounded-3xl p-[4px] overflow-hidden transition-all duration-500 group-hover:scale-105"
                style={{
                  background:
                    theme === "dark"
                      ? "linear-gradient(135deg, rgba(99, 102, 241, 0.6) 0%, rgba(168, 85, 247, 0.5) 25%, rgba(59, 130, 246, 0.4) 50%, rgba(168, 85, 247, 0.5) 75%, rgba(99, 102, 241, 0.6) 100%)"
                      : "linear-gradient(135deg, rgba(59, 130, 246, 0.5) 0%, rgba(147, 51, 234, 0.4) 25%, rgba(99, 102, 241, 0.3) 50%, rgba(147, 51, 234, 0.4) 75%, rgba(59, 130, 246, 0.5) 100%)",
                  boxShadow:
                    theme === "dark"
                      ? "inset 0 2px 10px rgba(99, 102, 241, 0.2), 0 0 40px rgba(99, 102, 241, 0.1)"
                      : "inset 0 2px 10px rgba(59, 130, 246, 0.15), 0 0 40px rgba(59, 130, 246, 0.08)",
                }}
              >
                <div className="w-full h-full rounded-2xl overflow-hidden bg-gradient-to-br from-white/5 to-black/5">
                  <Image
                    src={getProfileImage()}
                    alt="Hiep Tran - AI Engineer and Full-Stack Developer profile photo"
                    width={320}
                    height={320}
                    className="rounded-2xl shadow-2xl object-cover transition-all duration-500 group-hover:scale-110"
                    style={{
                      width: "100%",
                      height: "100%",
                      objectFit: "cover",
                    }}
                    priority
                    sizes="(max-width: 768px) 280px, 320px"
                    placeholder="blur"
                    blurDataURL="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAAIAAoDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAhEAACAQMDBQAAAAAAAAAAAAABAgMABAUGIWGRkbHB0f/EABUBAQEAAAAAAAAAAAAAAAAAAAMF/8QAGhEAAgIDAAAAAAAAAAAAAAAAAAECEgMRkf/aAAwDAQACEQMRAD8AltJagyeH0AthI5xdrLcNM91BF5pX2HaH9bcfaSXWGaRmknyJckliyjqTzSlT54b6bk+h0R//2Q=="
                  />
                </div>
              </div>

              {/* Tech Tags with hover animation */}
              {/* AI/ML Tag */}
              <div
                className="absolute top-12 -left-14 px-3 py-1.5 rounded-full text-xs font-medium backdrop-blur-md border shadow-lg transition-all duration-700 opacity-0 group-hover:opacity-100 transform -translate-x-4 translate-y-2 group-hover:translate-x-0 group-hover:translate-y-0 group-hover:scale-100 scale-90"
                style={{
                  backgroundColor:
                    theme === "dark"
                      ? "rgba(30, 41, 59, 0.95)"
                      : "rgba(248, 250, 252, 0.95)",
                  borderColor:
                    theme === "dark"
                      ? "rgba(168, 85, 247, 0.5)"
                      : "rgba(147, 51, 234, 0.4)",
                  color: theme === "dark" ? "#c084fc" : "#7c3aed",
                  boxShadow:
                    theme === "dark"
                      ? "0 8px 25px -5px rgba(168, 85, 247, 0.3)"
                      : "0 8px 25px -5px rgba(124, 58, 237, 0.2)",
                  transitionDelay: "0.2s",
                }}
              >
                🤖 AI/ML
              </div>

              {/* Coffee Tag */}
              <div
                className="absolute -top-6 left-1/2 transform -translate-x-1/2 px-3 py-1.5 rounded-full text-xs font-medium backdrop-blur-md border shadow-lg transition-all duration-700 opacity-0 group-hover:opacity-100 translate-y-[-8px] group-hover:translate-y-0 group-hover:scale-100 scale-90"
                style={{
                  backgroundColor:
                    theme === "dark"
                      ? "rgba(30, 41, 59, 0.95)"
                      : "rgba(248, 250, 252, 0.95)",
                  borderColor:
                    theme === "dark"
                      ? "rgba(217, 119, 6, 0.5)"
                      : "rgba(180, 83, 9, 0.4)",
                  color: theme === "dark" ? "#f59e0b" : "#b45309",
                  boxShadow:
                    theme === "dark"
                      ? "0 8px 25px -5px rgba(217, 119, 6, 0.3)"
                      : "0 8px 25px -5px rgba(180, 83, 9, 0.2)",
                  transitionDelay: "0.4s",
                }}
              >
                ☕ Coffee
              </div>

              {/* Code Tag */}
              <div
                className="absolute top-1/2 -right-12 transform -translate-y-1/2 px-3 py-1.5 rounded-full text-xs font-medium backdrop-blur-md border shadow-lg transition-all duration-700 opacity-0 group-hover:opacity-100 translate-x-4 translate-y-2 group-hover:translate-x-0 group-hover:translate-y-0 group-hover:scale-100 scale-90"
                style={{
                  backgroundColor:
                    theme === "dark"
                      ? "rgba(30, 41, 59, 0.95)"
                      : "rgba(248, 250, 252, 0.95)",
                  borderColor:
                    theme === "dark"
                      ? "rgba(34, 197, 94, 0.5)"
                      : "rgba(21, 128, 61, 0.4)",
                  color: theme === "dark" ? "#4ade80" : "#15803d",
                  boxShadow:
                    theme === "dark"
                      ? "0 8px 25px -5px rgba(34, 197, 94, 0.3)"
                      : "0 8px 25px -5px rgba(21, 128, 61, 0.2)",
                  transitionDelay: "0.6s",
                }}
              >
                💻 Code
              </div>

              {/* Enhanced Available Badge */}
              <div
                className="absolute -bottom-4 -right-4 rounded-2xl px-4 py-2.5 shadow-xl border backdrop-blur-md flex items-center gap-3 transition-all duration-300 hover:scale-105"
                style={{
                  backgroundColor:
                    theme === "dark"
                      ? "rgba(30, 41, 59, 0.9)"
                      : "rgba(248, 250, 252, 0.95)",
                  borderColor:
                    theme === "dark"
                      ? "rgba(99, 102, 241, 0.3)"
                      : "rgba(59, 130, 246, 0.3)",
                  boxShadow:
                    theme === "dark"
                      ? "0 10px 30px -5px rgba(0, 0, 0, 0.5), 0 0 20px rgba(99, 102, 241, 0.2)"
                      : "0 10px 30px -5px rgba(0, 0, 0, 0.2), 0 0 20px rgba(59, 130, 246, 0.15)",
                }}
                role="status"
                aria-label="Availability status"
              >
                <div className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-full animate-pulse"
                    style={{
                      background:
                        theme === "dark"
                          ? "linear-gradient(135deg, #34d399 0%, #10b981 100%)"
                          : "linear-gradient(135deg, #22c55e 0%, #16a34a 100%)",
                      boxShadow:
                        theme === "dark"
                          ? "0 0 10px rgba(52, 211, 153, 0.5)"
                          : "0 0 10px rgba(34, 197, 94, 0.4)",
                    }}
                    aria-hidden="true"
                  ></div>
                  <span
                    className="text-sm font-semibold tracking-wide"
                    style={{ color: theme === "dark" ? "#e2e8f0" : "#1e293b" }}
                  >
                    Available for work
                  </span>
                </div>
              </div>
            </div>
          </aside>

          {/* Right Content */}
          <header className="flex-1 text-center lg:text-left max-w-2xl lg:max-w-3xl order-2 lg:order-2">
            <div className="mb-6">
              {/* Greeting */}
              <span
                className="inline-block text-base sm:text-lg font-medium mb-4 px-4 py-2 rounded-full border"
                style={{
                  color:
                    theme === "dark"
                      ? "rgba(148, 163, 184, 0.9)"
                      : "rgba(71, 85, 105, 0.8)",
                  backgroundColor:
                    theme === "dark"
                      ? "rgba(30, 41, 59, 0.5)"
                      : "rgba(248, 250, 252, 0.8)",
                  borderColor:
                    theme === "dark"
                      ? "rgba(99, 102, 241, 0.2)"
                      : "rgba(59, 130, 246, 0.2)",
                  fontFamily:
                    "var(--font-geist-sans), Inter, system-ui, sans-serif",
                  fontWeight: "500",
                  letterSpacing: "-0.01em",
                }}
              >
                Hi! I&apos;m Hiep Tran 👋
              </span>

              {/* Main heading */}
              <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-5xl xl:text-6xl font-bold mb-6 leading-[1.1]">
                <div className="mb-2">
                  <span
                    style={{
                      color:
                        theme === "dark"
                          ? "rgba(241, 245, 249, 0.95)"
                          : "rgba(30, 41, 59, 0.9)",
                      fontFamily:
                        "var(--font-geist-sans), Inter, system-ui, sans-serif",
                      fontWeight: "700",
                      letterSpacing: "-0.025em",
                    }}
                  >
                    I&apos;m{" "}
                  </span>
                  <span
                    className="bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 bg-clip-text text-transparent font-bold"
                    style={{
                      backgroundImage:
                        theme === "dark"
                          ? "linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #818cf8 100%)"
                          : "linear-gradient(135deg, #2563eb 0%, #7c3aed 50%, #4f46e5 100%)",
                      fontFamily:
                        "var(--font-geist-sans), Inter, system-ui, sans-serif",
                      fontWeight: "700",
                      letterSpacing: "-0.03em",
                    }}
                    aria-live="polite"
                  >
                    {displayText}
                  </span>
                  <span
                    className="animate-blink border-r-2 ml-1 inline-block"
                    style={{
                      borderColor: theme === "dark" ? "#60a5fa" : "#2563eb",
                      height: "1em",
                      width: "3px",
                    }}
                    aria-hidden="true"
                  ></span>
                </div>
              </h1>
            </div>

            {/* Subtitle */}
            <h2
              className="text-lg sm:text-xl md:text-2xl lg:text-2xl xl:text-3xl font-medium mb-6 leading-relaxed"
              style={{
                color:
                  theme === "dark"
                    ? "rgba(203, 213, 225, 0.9)"
                    : "rgba(71, 85, 105, 0.9)",
                fontFamily:
                  "var(--font-geist-sans), Inter, system-ui, sans-serif",
                fontWeight: "500",
                letterSpacing: "-0.02em",
              }}
            >
              Building tomorrow&apos;s solutions with{" "}
              <span
                className="font-semibold"
                style={{
                  color: theme === "dark" ? "#60a5fa" : "#2563eb",
                  fontWeight: "600",
                }}
              >
                AI & Code
              </span>
            </h2>

            {/* Description */}
            <p
              className="text-base md:text-lg mb-8 leading-relaxed max-w-2xl mx-auto lg:mx-0"
              style={{
                color:
                  theme === "dark"
                    ? "rgba(148, 163, 184, 0.85)"
                    : "rgba(71, 85, 105, 0.8)",
                fontFamily:
                  "var(--font-geist-sans), Inter, system-ui, sans-serif",
                fontWeight: "400",
                letterSpacing: "-0.005em",
                lineHeight: "1.7",
              }}
            >
              Specialized in developing scalable AI systems, enterprise-grade
              applications, and innovative digital solutions. With expertise in
              Machine Learning, Deep Learning, and Full-Stack Development, I
              help businesses leverage technology to drive growth and achieve
              their strategic objectives.
            </p>

            <nav
              className="flex flex-col sm:flex-row gap-4 mb-10 justify-center lg:justify-start"
              aria-label="Main navigation buttons"
            >
              <Link
                href="/blog"
                className="group relative inline-flex items-center justify-center px-6 py-3 font-semibold rounded-lg transition-all duration-300 shadow-lg text-white gap-2 overflow-hidden text-sm sm:text-base"
                style={{
                  background:
                    theme === "dark"
                      ? "linear-gradient(135deg, #3b82f6 0%, #6366f1 50%, #8b5cf6 100%)"
                      : "linear-gradient(135deg, #2563eb 0%, #4f46e5 50%, #7c3aed 100%)",
                  boxShadow:
                    theme === "dark"
                      ? "0 8px 25px -5px rgba(99, 102, 241, 0.3), 0 4px 15px -3px rgba(99, 102, 241, 0.2)"
                      : "0 8px 25px -5px rgba(37, 99, 235, 0.25), 0 4px 15px -3px rgba(37, 99, 235, 0.15)",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform =
                    "translateY(-2px) scale(1.02)";
                  e.currentTarget.style.boxShadow =
                    theme === "dark"
                      ? "0 15px 35px -5px rgba(99, 102, 241, 0.4), 0 8px 20px -3px rgba(99, 102, 241, 0.3)"
                      : "0 15px 35px -5px rgba(37, 99, 235, 0.3), 0 8px 20px -3px rgba(37, 99, 235, 0.2)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = "translateY(0) scale(1)";
                  e.currentTarget.style.boxShadow =
                    theme === "dark"
                      ? "0 8px 25px -5px rgba(99, 102, 241, 0.3), 0 4px 15px -3px rgba(99, 102, 241, 0.2)"
                      : "0 8px 25px -5px rgba(37, 99, 235, 0.25), 0 4px 15px -3px rgba(37, 99, 235, 0.15)";
                }}
                aria-describedby="blog-description"
              >
                <span className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 transform -skew-x-12 group-hover:animate-shine"></span>
                <svg
                  className="w-5 h-5 relative z-10 group-hover:rotate-12 transition-transform duration-300"
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
                <span className="relative z-10">Explore Blog</span>
              </Link>

              <Link
                href="/projects"
                className="group relative inline-flex items-center justify-center px-6 py-3 font-semibold rounded-lg transition-all duration-300 border-2 gap-2 overflow-hidden backdrop-blur-sm text-sm sm:text-base"
                style={{
                  backgroundColor:
                    theme === "dark"
                      ? "rgba(30, 41, 59, 0.8)"
                      : "rgba(248, 250, 252, 0.9)",
                  borderColor:
                    theme === "dark"
                      ? "rgba(99, 102, 241, 0.3)"
                      : "rgba(59, 130, 246, 0.3)",
                  color: theme === "dark" ? "#e2e8f0" : "#1e293b",
                  boxShadow:
                    theme === "dark"
                      ? "0 6px 20px -5px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(99, 102, 241, 0.1)"
                      : "0 6px 20px -5px rgba(0, 0, 0, 0.1), inset 0 1px 0 rgba(59, 130, 246, 0.1)",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor =
                    theme === "dark"
                      ? "rgba(99, 102, 241, 0.1)"
                      : "rgba(59, 130, 246, 0.05)";
                  e.currentTarget.style.borderColor =
                    theme === "dark" ? "#60a5fa" : "#3b82f6";
                  e.currentTarget.style.color =
                    theme === "dark" ? "#60a5fa" : "#2563eb";
                  e.currentTarget.style.transform =
                    "translateY(-2px) scale(1.02)";
                  e.currentTarget.style.boxShadow =
                    theme === "dark"
                      ? "0 12px 30px -5px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(99, 102, 241, 0.2)"
                      : "0 12px 30px -5px rgba(0, 0, 0, 0.15), inset 0 1px 0 rgba(59, 130, 246, 0.2)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor =
                    theme === "dark"
                      ? "rgba(30, 41, 59, 0.8)"
                      : "rgba(248, 250, 252, 0.9)";
                  e.currentTarget.style.borderColor =
                    theme === "dark"
                      ? "rgba(99, 102, 241, 0.3)"
                      : "rgba(59, 130, 246, 0.3)";
                  e.currentTarget.style.color =
                    theme === "dark" ? "#e2e8f0" : "#1e293b";
                  e.currentTarget.style.transform = "translateY(0) scale(1)";
                  e.currentTarget.style.boxShadow =
                    theme === "dark"
                      ? "0 6px 20px -5px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(99, 102, 241, 0.1)"
                      : "0 6px 20px -5px rgba(0, 0, 0, 0.1), inset 0 1px 0 rgba(59, 130, 246, 0.1)";
                }}
                aria-describedby="projects-description"
              >
                <span className="absolute inset-0 bg-gradient-to-r from-transparent via-current/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></span>
                <svg
                  className="w-5 h-5 relative z-10 group-hover:rotate-12 transition-transform duration-300"
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

            {/* Enhanced Connect Section */}
            <div className="mb-8">
              <p
                className="mb-4 font-medium text-xs tracking-wide uppercase"
                style={{
                  color:
                    theme === "dark"
                      ? "rgba(148, 163, 184, 0.7)"
                      : "rgba(100, 116, 139, 0.7)",
                }}
              >
                Connect with me
              </p>
              <div
                className="flex gap-4 justify-center lg:justify-start"
                role="list"
                aria-label="Social media links"
              >
                <Link
                  href="https://github.com/hieptran1812"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="group p-2.5 rounded-lg transition-all duration-300 backdrop-blur-sm border"
                  style={{
                    backgroundColor:
                      theme === "dark"
                        ? "rgba(30, 41, 59, 0.5)"
                        : "rgba(248, 250, 252, 0.8)",
                    borderColor:
                      theme === "dark"
                        ? "rgba(99, 102, 241, 0.2)"
                        : "rgba(59, 130, 246, 0.2)",
                    color:
                      theme === "dark"
                        ? "rgba(148, 163, 184, 0.8)"
                        : "rgba(71, 85, 105, 0.8)",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor =
                      theme === "dark"
                        ? "rgba(99, 102, 241, 0.1)"
                        : "rgba(59, 130, 246, 0.05)";
                    e.currentTarget.style.borderColor =
                      theme === "dark" ? "#60a5fa" : "#3b82f6";
                    e.currentTarget.style.color =
                      theme === "dark" ? "#60a5fa" : "#2563eb";
                    e.currentTarget.style.transform =
                      "translateY(-1px) scale(1.05)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor =
                      theme === "dark"
                        ? "rgba(30, 41, 59, 0.5)"
                        : "rgba(248, 250, 252, 0.8)";
                    e.currentTarget.style.borderColor =
                      theme === "dark"
                        ? "rgba(99, 102, 241, 0.2)"
                        : "rgba(59, 130, 246, 0.2)";
                    e.currentTarget.style.color =
                      theme === "dark"
                        ? "rgba(148, 163, 184, 0.8)"
                        : "rgba(71, 85, 105, 0.8)";
                    e.currentTarget.style.transform = "translateY(0) scale(1)";
                  }}
                  aria-label="Visit Hiep Tran's GitHub profile"
                  role="listitem"
                >
                  <svg
                    className="w-5 h-5 group-hover:scale-110 transition-transform duration-300"
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
                  className="group p-2.5 rounded-lg transition-all duration-300 backdrop-blur-sm border"
                  style={{
                    backgroundColor:
                      theme === "dark"
                        ? "rgba(30, 41, 59, 0.5)"
                        : "rgba(248, 250, 252, 0.8)",
                    borderColor:
                      theme === "dark"
                        ? "rgba(99, 102, 241, 0.2)"
                        : "rgba(59, 130, 246, 0.2)",
                    color:
                      theme === "dark"
                        ? "rgba(148, 163, 184, 0.8)"
                        : "rgba(71, 85, 105, 0.8)",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor =
                      theme === "dark"
                        ? "rgba(99, 102, 241, 0.1)"
                        : "rgba(59, 130, 246, 0.05)";
                    e.currentTarget.style.borderColor =
                      theme === "dark" ? "#60a5fa" : "#3b82f6";
                    e.currentTarget.style.color =
                      theme === "dark" ? "#60a5fa" : "#2563eb";
                    e.currentTarget.style.transform =
                      "translateY(-1px) scale(1.05)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor =
                      theme === "dark"
                        ? "rgba(30, 41, 59, 0.5)"
                        : "rgba(248, 250, 252, 0.8)";
                    e.currentTarget.style.borderColor =
                      theme === "dark"
                        ? "rgba(99, 102, 241, 0.2)"
                        : "rgba(59, 130, 246, 0.2)";
                    e.currentTarget.style.color =
                      theme === "dark"
                        ? "rgba(148, 163, 184, 0.8)"
                        : "rgba(71, 85, 105, 0.8)";
                    e.currentTarget.style.transform = "translateY(0) scale(1)";
                  }}
                  aria-label="Visit Hiep Tran's LinkedIn profile"
                  role="listitem"
                >
                  <svg
                    className="w-5 h-5 group-hover:scale-110 transition-transform duration-300"
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
                  className="group p-2.5 rounded-lg transition-all duration-300 backdrop-blur-sm border"
                  style={{
                    backgroundColor:
                      theme === "dark"
                        ? "rgba(30, 41, 59, 0.5)"
                        : "rgba(248, 250, 252, 0.8)",
                    borderColor:
                      theme === "dark"
                        ? "rgba(99, 102, 241, 0.2)"
                        : "rgba(59, 130, 246, 0.2)",
                    color:
                      theme === "dark"
                        ? "rgba(148, 163, 184, 0.8)"
                        : "rgba(71, 85, 105, 0.8)",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor =
                      theme === "dark"
                        ? "rgba(99, 102, 241, 0.1)"
                        : "rgba(59, 130, 246, 0.05)";
                    e.currentTarget.style.borderColor =
                      theme === "dark" ? "#60a5fa" : "#3b82f6";
                    e.currentTarget.style.color =
                      theme === "dark" ? "#60a5fa" : "#2563eb";
                    e.currentTarget.style.transform =
                      "translateY(-1px) scale(1.05)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor =
                      theme === "dark"
                        ? "rgba(30, 41, 59, 0.5)"
                        : "rgba(248, 250, 252, 0.8)";
                    e.currentTarget.style.borderColor =
                      theme === "dark"
                        ? "rgba(99, 102, 241, 0.2)"
                        : "rgba(59, 130, 246, 0.2)";
                    e.currentTarget.style.color =
                      theme === "dark"
                        ? "rgba(148, 163, 184, 0.8)"
                        : "rgba(71, 85, 105, 0.8)";
                    e.currentTarget.style.transform = "translateY(0) scale(1)";
                  }}
                  aria-label="Follow Hiep Tran on X"
                  role="listitem"
                >
                  <svg
                    className="w-5 h-5 group-hover:scale-110 transition-transform duration-300"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                  >
                    <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                  </svg>
                </Link>

                <Link
                  href="mailto:hieptran.jobs@gmail.com"
                  className="group p-2.5 rounded-lg transition-all duration-300 backdrop-blur-sm border"
                  style={{
                    backgroundColor:
                      theme === "dark"
                        ? "rgba(30, 41, 59, 0.5)"
                        : "rgba(248, 250, 252, 0.8)",
                    borderColor:
                      theme === "dark"
                        ? "rgba(99, 102, 241, 0.2)"
                        : "rgba(59, 130, 246, 0.2)",
                    color:
                      theme === "dark"
                        ? "rgba(148, 163, 184, 0.8)"
                        : "rgba(71, 85, 105, 0.8)",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor =
                      theme === "dark"
                        ? "rgba(99, 102, 241, 0.1)"
                        : "rgba(59, 130, 246, 0.05)";
                    e.currentTarget.style.borderColor =
                      theme === "dark" ? "#60a5fa" : "#3b82f6";
                    e.currentTarget.style.color =
                      theme === "dark" ? "#60a5fa" : "#2563eb";
                    e.currentTarget.style.transform =
                      "translateY(-1px) scale(1.05)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor =
                      theme === "dark"
                        ? "rgba(30, 41, 59, 0.5)"
                        : "rgba(248, 250, 252, 0.8)";
                    e.currentTarget.style.borderColor =
                      theme === "dark"
                        ? "rgba(99, 102, 241, 0.2)"
                        : "rgba(59, 130, 246, 0.2)";
                    e.currentTarget.style.color =
                      theme === "dark"
                        ? "rgba(148, 163, 184, 0.8)"
                        : "rgba(71, 85, 105, 0.8)";
                    e.currentTarget.style.transform = "translateY(0) scale(1)";
                  }}
                  aria-label="Send email to Hiep Tran"
                  role="listitem"
                >
                  <svg
                    className="w-5 h-5 group-hover:scale-110 transition-transform duration-300"
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
          </header>
        </div>
      </div>
    </section>
  );
}
