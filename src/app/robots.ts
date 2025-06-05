import { MetadataRoute } from "next";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      {
        userAgent: "*",
        allow: "/",
        disallow: [
          "/private/",
          "/admin/",
          "/_next/",
          "/api/",
          "*.json",
          "/sitemap.xml",
          "/.well-known/",
          "/favicon.ico",
          "/robots.txt",
        ],
        crawlDelay: 1,
      },
      {
        userAgent: "Googlebot",
        allow: "/",
        disallow: ["/private/", "/admin/", "/_next/", "/api/"],
        crawlDelay: 1,
      },
      {
        userAgent: "Bingbot",
        allow: "/",
        disallow: ["/private/", "/admin/", "/_next/", "/api/"],
        crawlDelay: 2,
      },
      {
        userAgent: "GPTBot",
        disallow: "/",
      },
      {
        userAgent: "Google-Extended",
        disallow: "/",
      },
      {
        userAgent: "CCBot",
        disallow: "/",
      },
      {
        userAgent: "ChatGPT-User",
        disallow: "/",
      },
      {
        userAgent: "anthropic-ai",
        disallow: "/",
      },
      {
        userAgent: "Claude-Web",
        disallow: "/",
      },
    ],
    sitemap: "https://halleyverse.dev/sitemap.xml",
    host: "https://halleyverse.dev",
  };
}
