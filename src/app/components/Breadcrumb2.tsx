"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

interface BreadcrumbItem {
  label: string;
  href: string;
  isCurrentPage?: boolean;
}

interface BreadcrumbProps {
  items?: BreadcrumbItem[];
  className?: string;
}

const generateBreadcrumbItems = (pathname: string): BreadcrumbItem[] => {
  const segments = pathname
    .split("/")
    .filter(
      (segment) => segment !== "" && segment !== "page" && segment !== "layout"
    );

  const items: BreadcrumbItem[] = [
    {
      label: "Home",
      href: "/",
    },
  ];

  // Build breadcrumb items from URL segments
  let currentPath = "";
  segments.forEach((segment, index) => {
    currentPath += `/${segment}`;
    const isLast = index === segments.length - 1;

    // Skip certain system segments and file extensions
    const skipSegments = ["api", "_next", "static", "tsx", "js", "ts"];
    if (skipSegments.includes(segment) || segment.includes(".")) {
      return;
    }

    // Convert segment to readable label
    let label = segment.replace(/-/g, " ");
    label = label.charAt(0).toUpperCase() + label.slice(1);

    // Enhanced label mapping for better Vietnamese SEO and readability
    const labelMap: Record<string, string> = {
      "paper-reading": "Paper Reading",
      "machine-learning": "Machine Learning",
      "software-development": "Software Development",
      crypto: "Cryptocurrency",
      notes: "Notes",
      "modern-portfolio-nextjs": "Modern Portfolio Next.js",
      "open-source-guide": "Open Source Guide",
      about: "About",
      projects: "Projects",
      contact: "Contact",
      blog: "Blog",
      privacy: "Privacy Policy",
      terms: "Terms of Service",
      search: "Search",
    };

    if (labelMap[segment]) {
      label = labelMap[segment];
    }

    items.push({
      label,
      href: currentPath,
      isCurrentPage: isLast,
    });
  });

  return items;
};

export default function Breadcrumb({ items, className = "" }: BreadcrumbProps) {
  const pathname = usePathname();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Don't render until mounted to avoid hydration issues
  if (!mounted) {
    return <div className={`mb-6 ${className}`} style={{ height: "24px" }} />;
  }

  // Don't show breadcrumb on homepage
  if (pathname === "/") {
    return null;
  }

  const breadcrumbItems = items || generateBreadcrumbItems(pathname);

  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: breadcrumbItems.map((item, index) => ({
      "@type": "ListItem",
      position: index + 1,
      name: item.label,
      item: `https://hieptran.dev${item.href}`,
    })),
  };

  return (
    <>
      {/* JSON-LD structured data for breadcrumbs */}
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />

      <nav
        aria-label="Breadcrumb navigation"
        className={`mb-6 ${className}`}
        role="navigation"
      >
        <ol
          className="flex items-center space-x-2 text-sm"
          itemScope
          itemType="https://schema.org/BreadcrumbList"
        >
          {breadcrumbItems.map((item, index) => (
            <li
              key={`${item.href}-${index}`}
              className="flex items-center"
              itemProp="itemListElement"
              itemScope
              itemType="https://schema.org/ListItem"
            >
              {/* Hidden position for schema */}
              <meta itemProp="position" content={(index + 1).toString()} />

              {index > 0 && (
                <svg
                  className="w-4 h-4 mx-2 text-gray-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  aria-hidden="true"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M9 5l7 7-7 7"
                  />
                </svg>
              )}

              {item.isCurrentPage ? (
                <span
                  className="font-medium"
                  style={{ color: "var(--text-primary)" }}
                  aria-current="page"
                  itemProp="name"
                >
                  {item.label}
                </span>
              ) : (
                <Link
                  href={item.href}
                  className="transition-colors duration-200 hover:underline"
                  style={{ color: "var(--text-secondary)" }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.color = "var(--accent)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.color = "var(--text-secondary)";
                  }}
                  itemProp="item"
                >
                  <span itemProp="name">{item.label}</span>
                </Link>
              )}
            </li>
          ))}
        </ol>
      </nav>
    </>
  );
}
