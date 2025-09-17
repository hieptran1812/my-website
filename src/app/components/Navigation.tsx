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
    description: "View portfolio of AI and web development projects",
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
        name: "Crypto",
        href: "/blog/crypto",
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
      className={`fixed top-0 z-50 w-full transition-all duration-300 backdrop-blur-md ${
        isScrollingUp ? "translate-y-0" : "-translate-y-full"
      } ${
        isScrolled
          ? "navbar-glass-scrolled border-b shadow-lg"
          : "navbar-glass border-b"
      }`}
      style={{
        borderColor: "var(--border)",
        backgroundColor: isScrolled
          ? "var(--background)/95"
          : "var(--background)/80",
        backdropFilter: isScrolled ? "blur(20px)" : "blur(12px)",
        boxShadow: isScrolled ? "0 4px 20px var(--shadow)/10" : "none",
      }}
      role="navigation"
      aria-label="Main navigation"
    >
      <div className="container mx-auto flex items-center justify-between px-6 py-4">
        {/* Logo */}
        <Link
          href="/"
          className="flex items-center gap-3 font-bold transition-all duration-300 hover:scale-105"
          style={{ color: "var(--text-primary)" }}
          aria-label="Hiep Tran - Home"
          title="Go to homepage"
        >
          <div className="relative w-9 h-9 rounded-xl overflow-hidden shadow-lg transition-all duration-300 hover:shadow-xl">
            {mounted ? (
              <>
                {/* Light mode image */}
                <Image
                  src="/about-profile.webp"
                  alt="Hiep Tran Profile Light Mode"
                  width={36}
                  height={36}
                  className="absolute inset-0 w-full h-full object-cover transition-opacity duration-500"
                  style={{
                    opacity: theme === "light" ? 1 : 0,
                  }}
                  priority
                />
                {/* Dark mode image */}
                <Image
                  src="/about-profile.webp"
                  alt="Hiep Tran Profile Dark Mode"
                  width={36}
                  height={36}
                  className="absolute inset-0 w-full h-full object-cover transition-opacity duration-500"
                  style={{
                    opacity: theme === "dark" ? 1 : 0,
                  }}
                  priority
                />
              </>
            ) : (
              /* Fallback while loading */
              <div className="w-full h-full bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl flex items-center justify-center text-white font-bold text-sm">
                H
              </div>
            )}
          </div>
          <span className="nav-brand-text font-sans hidden sm:block text-lg tracking-tight font-semibold">
            Hiep Tran
          </span>
        </Link>

        {/* Desktop Navigation */}
        <div className="hidden lg:flex items-center gap-6" ref={dropdownRef}>
          {navLinks.map((link) => (
            <div key={link.name} className="relative">
              {link.hasDropdown ? (
                <div className="relative">
                  <button
                    onClick={() => handleDropdownToggle(link.name)}
                    className="navbar-button flex items-center gap-3 px-5 py-3 rounded-2xl transition-all duration-300 text-sm group font-semibold tracking-wide relative overflow-hidden backdrop-blur-sm"
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
                      fontWeight: "600",
                      fontSize: "0.875rem",
                      letterSpacing: "0.025em",
                      fontFamily:
                        "var(--font-geist-sans), -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
                      textShadow:
                        activeDropdown === link.name ||
                        pathname.startsWith(link.href)
                          ? "0 1px 2px rgba(0,0,0,0.1)"
                          : "none",
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.color = "var(--accent)";
                      e.currentTarget.style.backgroundColor =
                        "var(--surface-accent)";
                      e.currentTarget.style.transform =
                        "translateY(-2px) scale(1.02)";
                      e.currentTarget.style.boxShadow =
                        "0 8px 25px var(--shadow)/15, 0 0 0 1px var(--accent)/20";
                      e.currentTarget.style.textShadow =
                        "0 1px 2px rgba(0,0,0,0.1)";
                    }}
                    onMouseLeave={(e) => {
                      if (
                        activeDropdown !== link.name &&
                        !pathname.startsWith(link.href)
                      ) {
                        e.currentTarget.style.color = "var(--text-primary)";
                        e.currentTarget.style.backgroundColor = "transparent";
                        e.currentTarget.style.textShadow = "none";
                      }
                      e.currentTarget.style.transform =
                        "translateY(0px) scale(1)";
                      e.currentTarget.style.boxShadow = "none";
                    }}
                    aria-expanded={activeDropdown === link.name}
                    aria-haspopup="menu"
                    aria-label={`${link.name} menu - ${link.description}`}
                  >
                    {/* Enhanced active page underline for blog section */}
                    {pathname.startsWith(link.href) && (
                      <span className="absolute -bottom-2 left-1/2 -translate-x-1/2 w-full h-1 bg-gradient-to-r from-transparent via-[var(--accent)] to-transparent rounded-full shadow-lg shadow-[var(--accent)]/50"></span>
                    )}

                    {/* Ripple effect */}
                    <span className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent opacity-0 group-active:opacity-100 transition-opacity duration-200 transform -skew-x-12 group-active:animate-pulse"></span>

                    <span className="relative z-10 flex-shrink-0">
                      {link.icon}
                    </span>
                    <span className="relative z-10 font-semibold tracking-wide nav-text">
                      {link.name}
                    </span>
                    <svg
                      className={`relative z-10 w-4 h-4 transition-transform duration-200 ${
                        activeDropdown === link.name ? "rotate-180" : ""
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
                        <span className="absolute inset-0 rounded-2xl border-2 border-[var(--accent)]/30 animate-pulse"></span>
                      </>
                    )}
                  </button>

                  {/* Dropdown Menu */}
                  {activeDropdown === link.name && (
                    <div
                      className="navbar-dropdown absolute top-full right-0 mt-2 w-56 py-2 rounded-xl"
                      role="menu"
                      aria-label={`${link.name} submenu`}
                    >
                      {link.dropdownItems?.map((item) => (
                        <Link
                          key={item.name}
                          href={item.href}
                          onClick={() => setActiveDropdown(null)}
                          className="navbar-button flex items-center gap-3 px-4 py-3 text-sm relative overflow-hidden group rounded-lg mx-2 font-semibold tracking-wide backdrop-blur-sm"
                          style={{
                            color:
                              pathname === item.href
                                ? "var(--accent)"
                                : "var(--text-primary)",
                            backgroundColor:
                              pathname === item.href
                                ? "var(--surface-accent)"
                                : "transparent",
                            borderLeft:
                              pathname === item.href
                                ? "4px solid var(--accent)"
                                : "4px solid transparent",
                            fontSize: "0.875rem",
                            letterSpacing: "0.025em",
                            fontFamily:
                              "var(--font-geist-sans), -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
                            textShadow:
                              pathname === item.href
                                ? "0 1px 2px rgba(0,0,0,0.1)"
                                : "none",
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.color = "var(--accent)";
                            e.currentTarget.style.backgroundColor =
                              "var(--surface-accent)";
                            e.currentTarget.style.transform =
                              "translateX(6px) scale(1.02)";
                            e.currentTarget.style.borderLeft =
                              "4px solid var(--accent)";
                            e.currentTarget.style.textShadow =
                              "0 1px 2px rgba(0,0,0,0.1)";
                            e.currentTarget.style.boxShadow =
                              "0 4px 12px var(--shadow)/10";
                          }}
                          onMouseLeave={(e) => {
                            if (pathname !== item.href) {
                              e.currentTarget.style.color =
                                "var(--text-primary)";
                              e.currentTarget.style.backgroundColor =
                                "transparent";
                              e.currentTarget.style.borderLeft =
                                "4px solid transparent";
                              e.currentTarget.style.textShadow = "none";
                            }
                            e.currentTarget.style.transform =
                              "translateX(0px) scale(1)";
                            e.currentTarget.style.boxShadow = "none";
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
                              <span className="absolute bottom-0 left-1/2 -translate-x-1/2 w-3/4 h-1 bg-gradient-to-r from-transparent via-[var(--accent)] to-transparent rounded-full"></span>
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
                  className="flex items-center gap-3 px-5 py-3 rounded-2xl transition-all duration-300 text-sm group relative overflow-hidden font-semibold tracking-wide backdrop-blur-sm"
                  style={{
                    color:
                      pathname === link.href
                        ? "var(--accent)"
                        : "var(--text-primary)",
                    backgroundColor:
                      pathname === link.href
                        ? "var(--surface-accent)"
                        : "transparent",
                    fontWeight: "600",
                    fontSize: "0.875rem",
                    letterSpacing: "0.025em",
                    fontFamily:
                      "var(--font-geist-sans), -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
                    textShadow:
                      pathname === link.href
                        ? "0 1px 2px rgba(0,0,0,0.1)"
                        : "none",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.color = "var(--accent)";
                    e.currentTarget.style.backgroundColor =
                      "var(--surface-accent)";
                    e.currentTarget.style.transform =
                      "translateY(-2px) scale(1.02)";
                    e.currentTarget.style.boxShadow =
                      "0 8px 25px var(--shadow)/15, 0 0 0 1px var(--accent)/20";
                    e.currentTarget.style.textShadow =
                      "0 1px 2px rgba(0,0,0,0.1)";
                  }}
                  onMouseLeave={(e) => {
                    if (pathname !== link.href) {
                      e.currentTarget.style.color = "var(--text-primary)";
                      e.currentTarget.style.backgroundColor = "transparent";
                      e.currentTarget.style.textShadow = "none";
                    }
                    e.currentTarget.style.transform =
                      "translateY(0px) scale(1)";
                    e.currentTarget.style.boxShadow = "none";
                  }}
                  aria-label={link.description}
                  title={link.description}
                >
                  {/* Enhanced active page underline */}
                  {pathname === link.href && (
                    <span className="absolute -bottom-2 left-1/2 -translate-x-1/2 w-full h-1 bg-gradient-to-r from-transparent via-[var(--accent)] to-transparent rounded-full shadow-lg shadow-[var(--accent)]/50"></span>
                  )}

                  {/* Ripple effect */}
                  <span className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent opacity-0 group-active:opacity-100 transition-opacity duration-200 transform -skew-x-12 group-active:animate-pulse"></span>

                  <span className="relative z-10 flex-shrink-0">
                    {link.icon}
                  </span>
                  <span className="relative z-10 font-semibold nav-text">
                    {link.name}
                  </span>

                  {/* Active page indicators */}
                  {pathname === link.href && (
                    <>
                      <span className="absolute inset-0 rounded-2xl border-2 border-[var(--accent)]/30 animate-pulse"></span>
                    </>
                  )}
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
        className={`lg:hidden overflow-hidden transition-all duration-300 ease-in-out ${
          isMobileMenuOpen ? "max-h-96 opacity-100" : "max-h-0 opacity-0"
        }`}
      >
        <div className="mobile-menu px-6 py-4">
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
                      className="flex items-center gap-3 py-3 pl-6 text-sm transition-colors duration-200 relative font-medium"
                      style={{
                        color:
                          pathname === item.href
                            ? "var(--accent)"
                            : "var(--text-primary)",
                        fontWeight: pathname === item.href ? "600" : "500",
                        borderLeft:
                          pathname === item.href
                            ? "3px solid var(--accent)"
                            : "3px solid transparent",
                      }}
                      onClick={() => setIsMobileMenuOpen(false)}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.color = "var(--accent)";
                      }}
                      onMouseLeave={(e) => {
                        if (pathname !== item.href) {
                          e.currentTarget.style.color = "var(--text-primary)";
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
                  className="flex items-center gap-2 py-3 transition-colors duration-200 text-sm relative font-medium"
                  style={{
                    color:
                      pathname === link.href
                        ? "var(--accent)"
                        : "var(--text-primary)",
                    fontWeight: pathname === link.href ? "600" : "500",
                    borderLeft:
                      pathname === link.href
                        ? "3px solid var(--accent)"
                        : "3px solid transparent",
                  }}
                  onClick={() => setIsMobileMenuOpen(false)}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.color = "var(--accent)";
                  }}
                  onMouseLeave={(e) => {
                    if (pathname !== link.href) {
                      e.currentTarget.style.color = "var(--text-primary)";
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
