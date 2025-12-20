"use client";

import React, { useState, useRef, useEffect } from "react";
import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import ThemeToggle from "./ThemeToggle";
import { useTheme } from "../ThemeProvider";

const navLinks = [
  {
    name: "About",
    href: "/about",
    description: "Learn about Hiep Tran's background and experience",
    icon: (
      <svg
        className="w-4 h-4"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
        aria-hidden="true"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
          d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
        />
      </svg>
    ),
  },
  {
    name: "Projects",
    href: "/projects",
    description: "View portfolio of AI projects",
    icon: (
      <svg
        className="w-4 h-4"
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
    ),
  },
  {
    name: "Blog",
    href: "/blog",
    description:
      "Read blog posts about AI, machine learning, and software development",
    icon: (
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
          d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"
        />
      </svg>
    ),
    hasDropdown: true,
    dropdownItems: [
      {
        name: "Paper Reading",
        href: "/blog/paper-reading",
        icon: (
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
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
        ),
      },
      {
        name: "Notes",
        href: "/blog/notes",
        icon: (
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
              d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
            />
          </svg>
        ),
      },
      {
        name: "Software Development",
        href: "/blog/software-development",
        icon: (
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
              d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"
            />
          </svg>
        ),
      },
      {
        name: "Machine Learning",
        href: "/blog/machine-learning",
        icon: (
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
              d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
            />
          </svg>
        ),
      },
      {
        name: "Trading",
        href: "/blog/trading",
        icon: (
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
              d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
        ),
      },
    ],
  },
  {
    name: "Search",
    href: "/search",
    description: "Search through blog posts, projects, and content",
    icon: (
      <svg
        className="w-4 h-4"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
        aria-hidden="true"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
          d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
        />
      </svg>
    ),
  },
  {
    name: "Contact",
    href: "/contact",
    description: "Get in touch for collaboration opportunities",
    icon: (
      <svg
        className="w-4 h-4"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
        aria-hidden="true"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
          d="M3 8l7.89 4.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
        />
      </svg>
    ),
  },
];

export default function Navigation() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [activeDropdown, setActiveDropdown] = useState<string | null>(null);
  const [isScrolled, setIsScrolled] = useState(false);
  const [isScrollingUp, setIsScrollingUp] = useState(true);
  const [lastScrollY, setLastScrollY] = useState(0);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const pathname = usePathname();
  const { theme, mounted } = useTheme();

  // Handle scroll behavior for enhanced sticky navigation
  useEffect(() => {
    const handleScroll = () => {
      const currentScrollY = window.scrollY;

      // Set background blur effect when scrolled
      setIsScrolled(currentScrollY > 10);

      // Detect scroll direction
      if (currentScrollY > lastScrollY && currentScrollY > 100) {
        // Scrolling down - hide nav
        setIsScrollingUp(false);
      } else {
        // Scrolling up - show nav
        setIsScrollingUp(true);
      }

      setLastScrollY(currentScrollY);
    };

    // Throttle scroll events for better performance
    let ticking = false;
    const throttledHandleScroll = () => {
      if (!ticking) {
        requestAnimationFrame(() => {
          handleScroll();
          ticking = false;
        });
        ticking = true;
      }
    };

    window.addEventListener("scroll", throttledHandleScroll);
    return () => {
      window.removeEventListener("scroll", throttledHandleScroll);
    };
  }, [lastScrollY]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setActiveDropdown(null);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const handleDropdownToggle = (linkName: string) => {
    setActiveDropdown(activeDropdown === linkName ? null : linkName);
  };

  return (
    <nav
      className={`fixed top-0 z-50 w-full transition-all duration-500 ease-in-out ${
        isScrollingUp ? "translate-y-0" : "-translate-y-full"
      } ${
        isScrolled
          ? "navbar-glass-scrolled border-b shadow-2xl"
          : "navbar-glass border-b border-opacity-30"
      }`}
      style={{
        borderColor: "var(--border)",
        backgroundColor: isScrolled
          ? "var(--background)/98"
          : "var(--background)/85",
        backdropFilter: isScrolled
          ? "blur(24px) saturate(180%)"
          : "blur(16px) saturate(150%)",
        WebkitBackdropFilter: isScrolled
          ? "blur(24px) saturate(180%)"
          : "blur(16px) saturate(150%)",
        boxShadow: isScrolled
          ? "0 8px 32px rgba(0,0,0,0.12), 0 2px 8px rgba(0,0,0,0.08)"
          : "0 2px 8px rgba(0,0,0,0.04)",
      }}
      role="navigation"
      aria-label="Main navigation"
    >
      <div className="max-w-7xl mx-auto flex items-center justify-between px-4 sm:px-6 lg:px-8 py-3 lg:py-4">
        {/* Logo */}
        <Link
          href="/"
          className="flex items-center gap-3 font-bold transition-all duration-300 hover:scale-[1.02] group"
          style={{ color: "var(--text-primary)" }}
          aria-label="Hiep Tran - Home"
          title="Go to homepage"
        >
          <div className="relative w-10 h-10 rounded-2xl overflow-hidden shadow-lg transition-all duration-300 group-hover:shadow-2xl group-hover:rotate-2 ring-2 ring-white/10">
            {mounted ? (
              <>
                {/* Light mode image */}
                <Image
                  src="/about-profile.webp"
                  alt="Hiep Tran Profile Light Mode"
                  width={40}
                  height={40}
                  className="absolute inset-0 w-full h-full object-cover transition-all duration-500 group-hover:scale-110"
                  style={{
                    opacity: theme === "light" ? 1 : 0,
                  }}
                  priority
                />
                {/* Dark mode image */}
                <Image
                  src="/about-profile.webp"
                  alt="Hiep Tran Profile Dark Mode"
                  width={40}
                  height={40}
                  className="absolute inset-0 w-full h-full object-cover transition-all duration-500 group-hover:scale-110"
                  style={{
                    opacity: theme === "dark" ? 1 : 0,
                  }}
                  priority
                />
                {/* Hover overlay */}
                <div className="absolute inset-0 bg-gradient-to-br from-blue-400/20 to-purple-500/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              </>
            ) : (
              /* Fallback while loading */
              <div className="w-full h-full bg-gradient-to-br from-blue-500 via-purple-600 to-indigo-700 rounded-2xl flex items-center justify-center text-white font-bold text-base shadow-inner">
                H
              </div>
            )}
          </div>
          <div className="hidden sm:flex flex-col">
            <span className="nav-brand-text font-sans text-lg tracking-tight font-bold bg-gradient-to-r from-gray-900 to-gray-700 dark:from-white dark:to-gray-300 bg-clip-text text-transparent">
              Hiep Tran
            </span>
            <span className="text-xs font-medium text-gray-500 dark:text-gray-400 -mt-1 tracking-wide">
              AI Engineer
            </span>
          </div>
        </Link>

        {/* Desktop Navigation */}
        <div className="hidden lg:flex items-center gap-2" ref={dropdownRef}>
          {navLinks.map((link) => (
            <div key={link.name} className="relative">
              {link.hasDropdown ? (
                <div className="relative">
                  <button
                    onClick={() => handleDropdownToggle(link.name)}
                    className="navbar-button flex items-center gap-2 px-4 py-2.5 rounded-xl transition-all duration-300 text-sm group font-medium relative overflow-hidden backdrop-blur-sm hover:shadow-lg hover:scale-[1.02]"
                    style={{
                      color:
                        activeDropdown === link.name ||
                        pathname.startsWith(link.href)
                          ? "var(--accent)"
                          : "var(--text-primary)",
                      backgroundColor:
                        activeDropdown === link.name ||
                        pathname.startsWith(link.href)
                          ? "var(--surface-accent)"
                          : "transparent",
                      border:
                        activeDropdown === link.name ||
                        pathname.startsWith(link.href)
                          ? "1px solid var(--accent)/20"
                          : "1px solid transparent",
                      fontWeight: "500",
                      fontSize: "0.875rem",
                      letterSpacing: "0.015em",
                      fontFamily:
                        "var(--font-geist-sans), -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.color = "var(--accent)";
                      e.currentTarget.style.backgroundColor =
                        "var(--surface-accent)";
                      e.currentTarget.style.borderColor = "var(--accent)/20";
                    }}
                    onMouseLeave={(e) => {
                      if (
                        activeDropdown !== link.name &&
                        !pathname.startsWith(link.href)
                      ) {
                        e.currentTarget.style.color = "var(--text-primary)";
                        e.currentTarget.style.backgroundColor = "transparent";
                        e.currentTarget.style.borderColor = "transparent";
                      }
                    }}
                    aria-expanded={activeDropdown === link.name}
                    aria-haspopup="menu"
                    aria-label={`${link.name} menu - ${link.description}`}
                  >
                    {/* Enhanced active page indicator */}
                    {pathname.startsWith(link.href) && (
                      <>
                        <span className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-4/5 h-0.5 bg-gradient-to-r from-transparent via-[var(--accent)] to-transparent rounded-full"></span>
                        <span className="absolute inset-0 bg-gradient-to-r from-[var(--accent)]/5 to-[var(--accent)]/10 rounded-xl"></span>
                      </>
                    )}

                    {/* Hover glow effect */}
                    <span className="absolute inset-0 bg-gradient-to-r from-[var(--accent)]/0 via-[var(--accent)]/5 to-[var(--accent)]/0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-xl"></span>

                    <span className="relative z-10 flex-shrink-0">
                      {link.icon}
                    </span>
                    <span className="relative z-10 font-semibold tracking-wide nav-text">
                      {link.name}
                    </span>
                    <svg
                      className={`relative z-10 w-4 h-4 transition-all duration-300 ${
                        activeDropdown === link.name
                          ? "rotate-180 scale-110"
                          : ""
                      }`}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M19 9l-7 7-7-7"
                      />
                    </svg>

                    {/* Active page indicators */}
                    {pathname.startsWith(link.href) && (
                      <>
                        <span className="absolute inset-0 rounded-xl bg-gradient-to-r from-[var(--accent)]/5 to-[var(--accent)]/10"></span>
                        <span className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-4/5 h-0.5 bg-gradient-to-r from-transparent via-[var(--accent)] to-transparent rounded-full"></span>
                      </>
                    )}

                    {/* Hover glow effect */}
                    <span className="absolute inset-0 bg-gradient-to-r from-[var(--accent)]/0 via-[var(--accent)]/5 to-[var(--accent)]/0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-xl"></span>
                  </button>

                  {/* Dropdown Menu */}
                  {activeDropdown === link.name && (
                    <div
                      className="navbar-dropdown absolute top-full right-0 mt-3 w-64 py-3 rounded-2xl shadow-2xl border backdrop-blur-xl"
                      style={{
                        backgroundColor: "var(--background)",
                        borderColor: "var(--border)",
                        boxShadow:
                          "0 20px 25px -5px rgba(0, 0, 0, 0.15), 0 10px 10px -5px rgba(0, 0, 0, 0.08), 0 0 0 1px var(--border)",
                      }}
                      role="menu"
                      aria-label={`${link.name} submenu`}
                    >
                      {link.dropdownItems?.map((item) => (
                        <Link
                          key={item.name}
                          href={item.href}
                          onClick={() => setActiveDropdown(null)}
                          className="navbar-button flex items-center gap-3 px-4 py-3 text-sm relative overflow-hidden group rounded-xl mx-2 font-medium transition-all duration-200 hover:scale-[1.02]"
                          style={{
                            color:
                              pathname === item.href
                                ? "var(--accent)"
                                : "var(--text-primary)",
                            backgroundColor:
                              pathname === item.href
                                ? "var(--surface-accent)"
                                : "transparent",
                            border:
                              pathname === item.href
                                ? "1px solid var(--accent)/20"
                                : "1px solid transparent",
                            fontSize: "0.875rem",
                            letterSpacing: "0.015em",
                            fontFamily:
                              "var(--font-geist-sans), -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.color = "var(--accent)";
                            e.currentTarget.style.backgroundColor =
                              "var(--surface-accent)";
                            e.currentTarget.style.borderColor =
                              "var(--accent)/20";
                          }}
                          onMouseLeave={(e) => {
                            if (pathname !== item.href) {
                              e.currentTarget.style.color =
                                "var(--text-primary)";
                              e.currentTarget.style.backgroundColor =
                                "transparent";
                              e.currentTarget.style.borderColor = "transparent";
                            }
                          }}
                          role="menuitem"
                          aria-label={`Visit ${item.name} section`}
                        >
                          <span className="flex-shrink-0">{item.icon}</span>
                          <span className="font-semibold tracking-wide nav-text">
                            {item.name}
                          </span>
                          {pathname === item.href && (
                            <>
                              <span className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-3/4 h-0.5 bg-gradient-to-r from-transparent via-[var(--accent)] to-transparent rounded-full"></span>
                              <span className="absolute inset-0 bg-gradient-to-r from-[var(--accent)]/5 to-[var(--accent)]/10 rounded-xl"></span>
                            </>
                          )}
                        </Link>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <Link
                  href={link.href}
                  className="flex items-center gap-2 px-4 py-2.5 rounded-xl transition-all duration-300 text-sm group relative overflow-hidden font-medium backdrop-blur-sm hover:shadow-lg hover:scale-[1.02]"
                  style={{
                    color:
                      pathname === link.href
                        ? "var(--accent)"
                        : "var(--text-primary)",
                    backgroundColor:
                      pathname === link.href
                        ? "var(--surface-accent)"
                        : "transparent",
                    border:
                      pathname === link.href
                        ? "1px solid var(--accent)/20"
                        : "1px solid transparent",
                    fontWeight: "500",
                    fontSize: "0.875rem",
                    letterSpacing: "0.015em",
                    fontFamily:
                      "var(--font-geist-sans), -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.color = "var(--accent)";
                    e.currentTarget.style.backgroundColor =
                      "var(--surface-accent)";
                    e.currentTarget.style.borderColor = "var(--accent)/20";
                  }}
                  onMouseLeave={(e) => {
                    if (pathname !== link.href) {
                      e.currentTarget.style.color = "var(--text-primary)";
                      e.currentTarget.style.backgroundColor = "transparent";
                      e.currentTarget.style.borderColor = "transparent";
                    }
                  }}
                  aria-label={link.description}
                  title={link.description}
                >
                  {/* Enhanced active page indicator */}
                  {pathname === link.href && (
                    <>
                      <span className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-4/5 h-0.5 bg-gradient-to-r from-transparent via-[var(--accent)] to-transparent rounded-full"></span>
                      <span className="absolute inset-0 bg-gradient-to-r from-[var(--accent)]/5 to-[var(--accent)]/10 rounded-xl"></span>
                    </>
                  )}

                  {/* Hover glow effect */}
                  <span className="absolute inset-0 bg-gradient-to-r from-[var(--accent)]/0 via-[var(--accent)]/5 to-[var(--accent)]/0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-xl"></span>

                  <span className="relative z-10 flex-shrink-0">
                    {link.icon}
                  </span>
                  <span className="relative z-10 font-semibold nav-text">
                    {link.name}
                  </span>

                  {/* Active page indicators */}
                  {pathname === link.href && (
                    <>
                      <span className="absolute inset-0 rounded-xl bg-gradient-to-r from-[var(--accent)]/5 to-[var(--accent)]/10"></span>
                      <span className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-4/5 h-0.5 bg-gradient-to-r from-transparent via-[var(--accent)] to-transparent rounded-full"></span>
                    </>
                  )}

                  {/* Hover glow effect */}
                  <span className="absolute inset-0 bg-gradient-to-r from-[var(--accent)]/0 via-[var(--accent)]/5 to-[var(--accent)]/0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-xl"></span>
                </Link>
              )}
            </div>
          ))}
        </div>

        {/* Right side - Theme Toggle, Mobile Menu */}
        <div className="flex items-center gap-2">
          <ThemeToggle />

          {/* Mobile Menu Toggle Button */}
          <button
            className="lg:hidden transition-colors duration-200"
            style={{ color: "var(--text-muted)" }}
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            aria-label="Toggle mobile menu"
            aria-expanded={isMobileMenuOpen}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = "var(--accent)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = "var(--text-muted)";
            }}
          >
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d={
                  isMobileMenuOpen
                    ? "M6 18L18 6M6 6l12 12"
                    : "M4 6h16M4 12h16m-7 6h7"
                }
              ></path>
            </svg>
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      <div
        className={`lg:hidden overflow-hidden transition-all duration-500 ease-in-out ${
          isMobileMenuOpen ? "max-h-[32rem] opacity-100" : "max-h-0 opacity-0"
        }`}
        style={{
          backgroundColor: "var(--background)/98",
          backdropFilter: "blur(20px)",
          borderTop: isMobileMenuOpen ? "1px solid var(--border)" : "none",
        }}
      >
        <div className="mobile-menu px-4 py-6 space-y-2">
          {navLinks.map((link) => (
            <div key={link.name}>
              {link.hasDropdown ? (
                <div>
                  <div
                    className="flex items-center gap-2 py-2 text-sm font-medium"
                    style={{ color: "var(--text-primary)" }}
                  >
                    {link.icon}
                    <span>{link.name}</span>
                  </div>
                  {link.dropdownItems?.map((item) => (
                    <Link
                      key={item.name}
                      href={item.href}
                      className="flex items-center gap-3 py-3 pl-6 text-sm transition-all duration-200 relative font-medium rounded-lg ml-4 mr-2 hover:scale-[1.02]"
                      style={{
                        color:
                          pathname === item.href
                            ? "var(--accent)"
                            : "var(--text-primary)",
                        backgroundColor:
                          pathname === item.href
                            ? "var(--surface-accent)"
                            : "transparent",
                        fontWeight: pathname === item.href ? "600" : "500",
                        borderLeft:
                          pathname === item.href
                            ? "3px solid var(--accent)"
                            : "3px solid transparent",
                      }}
                      onClick={() => setIsMobileMenuOpen(false)}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.color = "var(--accent)";
                        e.currentTarget.style.backgroundColor =
                          "var(--surface-accent)";
                      }}
                      onMouseLeave={(e) => {
                        if (pathname !== item.href) {
                          e.currentTarget.style.color = "var(--text-primary)";
                          e.currentTarget.style.backgroundColor = "transparent";
                        }
                      }}
                    >
                      {item.icon}
                      <span className="tracking-wide">{item.name}</span>
                      {pathname === item.href && (
                        <>
                          <span className="absolute left-0 ml-2 h-full w-1 bg-[var(--accent)]"></span>
                        </>
                      )}
                    </Link>
                  ))}
                </div>
              ) : (
                <Link
                  href={link.href}
                  className="flex items-center gap-3 py-3 px-4 transition-all duration-200 text-sm relative font-medium rounded-lg mx-2 hover:scale-[1.02]"
                  style={{
                    color:
                      pathname === link.href
                        ? "var(--accent)"
                        : "var(--text-primary)",
                    backgroundColor:
                      pathname === link.href
                        ? "var(--surface-accent)"
                        : "transparent",
                    fontWeight: pathname === link.href ? "600" : "500",
                    border:
                      pathname === link.href
                        ? "1px solid var(--accent)/20"
                        : "1px solid transparent",
                  }}
                  onClick={() => setIsMobileMenuOpen(false)}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.color = "var(--accent)";
                    e.currentTarget.style.backgroundColor =
                      "var(--surface-accent)";
                    e.currentTarget.style.borderColor = "var(--accent)/20";
                  }}
                  onMouseLeave={(e) => {
                    if (pathname !== link.href) {
                      e.currentTarget.style.color = "var(--text-primary)";
                      e.currentTarget.style.backgroundColor = "transparent";
                      e.currentTarget.style.borderColor = "transparent";
                    }
                  }}
                >
                  {link.icon}
                  <span className="tracking-wide">{link.name}</span>
                  {pathname === link.href && (
                    <>
                      <span className="absolute left-0 -ml-2 h-full w-1 bg-[var(--accent)]"></span>
                    </>
                  )}
                </Link>
              )}
            </div>
          ))}
        </div>
      </div>
    </nav>
  );
}
