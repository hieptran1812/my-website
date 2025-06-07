"use client";

import React from "react";
import Link from "next/link";
import Image from "next/image";

const navigationLinks = [
  { name: "Home", href: "/" },
  { name: "About", href: "/about" },
  { name: "Projects", href: "/projects" },
  { name: "Blog", href: "/blog" },
  { name: "Contact", href: "/contact" },
];

const blogCategories = [
  { name: "Paper Reading", href: "/blog/paper-reading" },
  { name: "Notes", href: "/blog/notes" },
  { name: "Software Development", href: "/blog/software-development" },
  { name: "Machine Learning", href: "/blog/machine-learning" },
  { name: "Crypto", href: "/blog/crypto" },
];

const resourceLinks = [
  { name: "Resume/CV", href: "/resume.pdf" },
  // { name: "Research Papers", href: "/research" },
];

const legalLinks = [
  { name: "Privacy Policy", href: "/privacy" },
  { name: "Terms of Service", href: "/terms" },
  // { name: "RSS Feed", href: "/blog/rss.xml" },
  // { name: "Sitemap", href: "/sitemap.xml" },
];

const socialLinks = [
  {
    name: "GitHub",
    href: "https://github.com/hieptran1812",
    icon: (
      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
        <path d="M12 0C5.374 0 0 5.373 0 12 0 17.302 3.438 21.8 8.207 23.387c.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0112 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z" />
      </svg>
    ),
  },
  {
    name: "LinkedIn",
    href: "https://www.linkedin.com/in/hieptran01",
    icon: (
      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
        <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
      </svg>
    ),
  },
  {
    name: "X",
    href: "https://x.com/halleytran01",
    icon: (
      <span className="relative group">
        <Image
          src="/x.svg"
          alt="X logo"
          width={20}
          height={20}
          className="w-5 h-5 transition-all duration-200"
          style={{ filter: "invert(0) grayscale(1)" }}
          priority={false}
        />
        <style jsx>{`
          .group:hover img {
            filter: invert(0.5) sepia(1) saturate(10) hue-rotate(180deg) !important;
          }
        `}</style>
      </span>
    ),
  },
  // {
  //   name: "Stack Overflow",
  //   href: "https://stackoverflow.com/users/hieptran1812",
  //   icon: (
  //     <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
  //       <path d="M15.725 0l-1.72 1.277 6.39 8.588 1.716-1.277L15.725 0zm-3.94 3.418l-1.369 1.644 8.225 6.85 1.369-1.644-8.225-6.85zm-3.15 4.465l-.905 1.94 9.702 4.517.904-1.94-9.701-4.517zm-1.85 4.86l-.44 2.093 10.473 2.201.44-2.092-10.473-2.203zM1.89 15.47V24h19.19v-8.53h-2.133v6.397H4.021v-6.396H1.89zm4.265 2.133v2.13h10.66v-2.13H6.154Z" />
  //     </svg>
  //   ),
  // },
  // {
  //   name: "YouTube",
  //   href: "https://youtube.com/@hieptran1812",
  //   icon: (
  //     <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
  //       <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z" />
  //     </svg>
  //   ),
  // },
  {
    name: "Email",
    href: "mailto:hieptran.jobs@gmail.com",
    icon: (
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
          d="M3 8l7.89 4.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
        />
      </svg>
    ),
  },
];

export default function Footer() {
  const currentYear = new Date().getFullYear();

  const organizationSchema = {
    "@context": "https://schema.org",
    "@type": "Person",
    name: "Hiep Tran",
    url: "https://halleyverse.dev",
    sameAs: [
      "https://github.com/hieptran1812",
      "https://www.linkedin.com/in/hieptran01",
      "https://x.com/halleytran01",
      "https://stackoverflow.com/users/hieptran1812",
      "https://youtube.com/@hieptran1812",
    ],
    jobTitle: "AI Engineer",
    workLocation: {
      "@type": "Place",
      address: {
        "@type": "PostalAddress",
        addressLocality: "Hanoi",
        addressCountry: "VN",
      },
    },
    email: "hieptran.jobs@gmail.com",
    knowsAbout: [
      "Artificial Intelligence",
      "Machine Learning",
      "Software Engineering",
      "Full-Stack Development",
      "Research",
    ],
  };

  return (
    <>
      {/* JSON-LD structured data for footer */}
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(organizationSchema) }}
      />

      <footer
        className="border-t transition-colors duration-300"
        role="contentinfo"
        aria-label="Site footer"
        style={{
          backgroundColor: "var(--background)",
          borderColor: "var(--border)",
        }}
      >
        {/* Compact Main Footer Content */}
        <div className="container mx-auto px-4 sm:px-6 py-8 sm:py-12 max-w-7xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 lg:gap-8">
            {/* Brand & Contact Section */}
            <div className="sm:col-span-2 lg:col-span-1">
              <div className="flex items-center gap-3 mb-4">
                <div
                  className="w-8 h-8 rounded-lg overflow-hidden shadow-lg"
                  style={{
                    boxShadow: "0 2px 8px var(--accent)/20",
                  }}
                >
                  <Image
                    src="/about-profile.png"
                    alt="Hiep Tran"
                    width={32}
                    height={32}
                    className="w-full h-full object-cover"
                  />
                </div>
                <h3
                  className="text-lg font-bold"
                  style={{ color: "var(--text-primary)" }}
                >
                  Hiep Tran
                </h3>
              </div>
              <p
                className="text-sm leading-relaxed mb-4"
                style={{ color: "var(--text-secondary)" }}
              >
                AI Engineer building intelligent systems.
              </p>

              {/* Contact Info */}
              <div className="space-y-2 mb-4">
                <div className="flex items-center gap-2 text-sm">
                  <svg
                    className="w-3 h-3 flex-shrink-0"
                    fill="currentColor"
                    viewBox="0 0 24 24"
                    style={{ color: "var(--accent)" }}
                  >
                    <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z" />
                  </svg>
                  <span style={{ color: "var(--text-secondary)" }}>
                    Hanoi, Vietnam
                  </span>
                </div>
                <div className="flex items-center gap-2 text-sm">
                  <svg
                    className="w-3 h-3 flex-shrink-0"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    style={{ color: "var(--accent)" }}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M3 8l7.89 4.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                    />
                  </svg>
                  <a
                    href="mailto:hieptran.jobs@gmail.com"
                    className="hover:underline break-all"
                    style={{ color: "var(--text-secondary)" }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.color = "var(--accent)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.color = "var(--text-secondary)";
                    }}
                  >
                    hieptran.jobs@gmail.com
                  </a>
                </div>
              </div>

              {/* Social Links */}
              <div className="flex flex-wrap gap-2">
                {socialLinks.map((link) => (
                  <a
                    key={link.name}
                    href={link.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="p-1.5 rounded border transition-all duration-200 hover:scale-110"
                    style={{
                      backgroundColor: "var(--card-bg)",
                      borderColor: "var(--card-border)",
                      color: "var(--text-secondary)",
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor =
                        "var(--surface-accent)";
                      e.currentTarget.style.color = "var(--accent)";
                      e.currentTarget.style.borderColor = "var(--accent)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = "var(--card-bg)";
                      e.currentTarget.style.color = "var(--text-secondary)";
                      e.currentTarget.style.borderColor = "var(--card-border)";
                    }}
                    title={link.name}
                  >
                    {link.icon}
                  </a>
                ))}
              </div>
            </div>

            {/* Quick Links */}
            <div>
              <h4
                className="font-semibold mb-3 text-sm"
                style={{ color: "var(--text-primary)" }}
              >
                Quick Links
              </h4>
              <ul className="space-y-2">
                {navigationLinks.map((link) => (
                  <li key={link.name}>
                    <Link
                      href={link.href}
                      className="text-sm transition-colors duration-200 hover:translate-x-1 inline-block"
                      style={{ color: "var(--text-secondary)" }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.color = "var(--accent)";
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.color = "var(--text-secondary)";
                      }}
                    >
                      {link.name}
                    </Link>
                  </li>
                ))}
                {resourceLinks.slice(0, 2).map((resource) => (
                  <li key={resource.name}>
                    <a
                      href={resource.href}
                      target={
                        resource.href.startsWith("http") ? "_blank" : undefined
                      }
                      rel={
                        resource.href.startsWith("http")
                          ? "noopener noreferrer"
                          : undefined
                      }
                      className="text-sm transition-colors duration-200 hover:translate-x-1 inline-block"
                      style={{ color: "var(--text-secondary)" }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.color = "var(--accent)";
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.color = "var(--text-secondary)";
                      }}
                    >
                      {resource.name}
                    </a>
                  </li>
                ))}
              </ul>
            </div>

            {/* Blog Categories */}
            <div>
              <h4
                className="font-semibold mb-3 text-sm"
                style={{ color: "var(--text-primary)" }}
              >
                Blog Topics
              </h4>
              <ul className="space-y-2">
                {blogCategories.slice(0, 5).map((category) => (
                  <li key={category.name}>
                    <Link
                      href={category.href}
                      className="text-sm transition-colors duration-200 hover:translate-x-1 inline-block"
                      style={{ color: "var(--text-secondary)" }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.color = "var(--accent)";
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.color = "var(--text-secondary)";
                      }}
                    >
                      {category.name}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>

            {/* Legal & Policies */}
            <div>
              <h4
                className="font-semibold mb-3 text-sm"
                style={{ color: "var(--text-primary)" }}
              >
                Legal & More
              </h4>
              <ul className="space-y-2">
                {legalLinks.map((link) => (
                  <li key={link.name}>
                    <Link
                      href={link.href}
                      target={link.href.endsWith(".xml") ? "_blank" : undefined}
                      rel={
                        link.href.endsWith(".xml")
                          ? "noopener noreferrer"
                          : undefined
                      }
                      className="text-sm transition-colors duration-200 hover:translate-x-1 inline-block"
                      style={{ color: "var(--text-secondary)" }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.color = "var(--accent)";
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.color = "var(--text-secondary)";
                      }}
                    >
                      {link.name}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Compact Newsletter Signup */}
          <div
            className="mt-6 sm:mt-8 p-4 rounded-xl border"
            style={{
              backgroundColor: "var(--card-bg)",
              borderColor: "var(--card-border)",
            }}
          >
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div className="w-full sm:w-auto">
                <h4
                  className="font-semibold text-sm mb-1"
                  style={{ color: "var(--text-primary)" }}
                >
                  Stay Updated
                </h4>
                <p
                  className="text-xs"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Get notified about new posts and updates
                </p>
              </div>
              <div className="flex flex-col sm:flex-row gap-2 w-full sm:w-auto">
                <input
                  type="email"
                  placeholder="your@email.com"
                  className="w-full sm:w-48 px-3 py-2 rounded-lg border text-sm transition-colors duration-200 focus:outline-none focus:ring-2"
                  style={{
                    backgroundColor: "var(--background)",
                    borderColor: "var(--border)",
                    color: "var(--text-primary)",
                  }}
                  onFocus={(e) => {
                    e.currentTarget.style.borderColor = "var(--accent)";
                    e.currentTarget.style.boxShadow =
                      "0 0 0 2px var(--accent)/20";
                  }}
                  onBlur={(e) => {
                    e.currentTarget.style.borderColor = "var(--border)";
                    e.currentTarget.style.boxShadow = "none";
                  }}
                />
                <button
                  className="px-4 py-2 rounded-lg font-medium text-sm transition-all duration-200 hover:scale-105 text-white w-full sm:w-auto"
                  style={{ backgroundColor: "var(--accent)" }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor =
                      "var(--accent-hover)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = "var(--accent)";
                  }}
                >
                  Subscribe
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Compact Bottom Section */}
        <div
          className="border-t py-4"
          style={{
            backgroundColor: "var(--surface)",
            borderColor: "var(--border)",
          }}
        >
          <div className="container mx-auto px-4 sm:px-6 max-w-7xl">
            <div className="flex flex-col lg:flex-row justify-between items-center gap-3 text-xs">
              <div className="flex flex-col sm:flex-row items-center gap-3">
                <p style={{ color: "var(--text-muted)" }}>
                  © {currentYear} Hiep Tran. All rights reserved.
                </p>
                <div className="flex items-center gap-3 text-center">
                  <Link
                    href="/privacy"
                    className="transition-colors duration-200"
                    style={{ color: "var(--text-muted)" }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.color = "var(--accent)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.color = "var(--text-muted)";
                    }}
                  >
                    Privacy
                  </Link>
                  <span style={{ color: "var(--text-muted)" }}>•</span>
                  <Link
                    href="/terms"
                    className="transition-colors duration-200"
                    style={{ color: "var(--text-muted)" }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.color = "var(--accent)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.color = "var(--text-muted)";
                    }}
                  >
                    Terms
                  </Link>
                  <span style={{ color: "var(--text-muted)" }}>•</span>
                  <Link
                    href="/sitemap.xml"
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
                  >
                    Sitemap
                  </Link>
                </div>
              </div>

              <div className="flex flex-col sm:flex-row items-center gap-3">
                <p
                  style={{ color: "var(--text-muted)" }}
                  className="text-center"
                >
                  Built with 💙
                </p>
                <div className="flex items-center gap-1">
                  <div
                    className="w-1.5 h-1.5 rounded-full animate-pulse"
                    style={{ backgroundColor: "var(--accent)" }}
                  ></div>
                  <span style={{ color: "var(--text-muted)" }}>Online</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </footer>
    </>
  );
}
