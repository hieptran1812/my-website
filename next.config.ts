import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  trailingSlash: true,
  images: {
    formats: ["image/avif", "image/webp"],
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048],
    imageSizes: [16, 32, 48, 64, 96, 128, 256],
    minimumCacheTTL: 31536000, // Cache for 1 year
    dangerouslyAllowSVG: true,
    contentDispositionType: "inline",
    contentSecurityPolicy: "default-src 'self'; script-src 'none'; sandbox;",
  },
  experimental: {
    // gray-matter is server-only — tree-shake hint does nothing for it. Keep
    // only the client-loaded packages so the optimiser stays useful.
    optimizePackageImports: ["react-icons", "d3", "katex"],
    scrollRestoration: true,
  },
  // Ship the build-time blog index with the listing routes that read it at
  // runtime (the loader falls back to a live corpus walk if it's ever absent).
  outputFileTracingIncludes: {
    "/api/blog/posts": ["./src/lib/generated/blogPostsIndex.json"],
    "/api/blog": ["./src/lib/generated/blogPostsIndex.json"],
  },
  compress: true,
  poweredByHeader: false,
  reactStrictMode: true,
  async headers() {
    const cspValue =
      "connect-src 'self' https://vitals.vercel-insights.com https://vercel-insights.com" +
        " https://cdn.jsdelivr.net;" +
      " script-src 'self' 'unsafe-inline' 'unsafe-eval' https://vercel.live blob:;" +
      " worker-src 'self' blob:;";
    return [
      {
        source: "/_next/static/(.*)",
        headers: [
          { key: "Cache-Control", value: "public, max-age=31536000, immutable" },
          { key: "Content-Security-Policy", value: cspValue },
        ],
      },
      {
        source: "/imgs/(.*)",
        headers: [
          { key: "Cache-Control", value: "public, max-age=2592000, stale-while-revalidate=86400" },
          { key: "Content-Security-Policy", value: cspValue },
        ],
      },
      {
        source: "/(.*).webp",
        headers: [
          { key: "Cache-Control", value: "public, max-age=2592000, stale-while-revalidate=86400" },
          { key: "Content-Security-Policy", value: cspValue },
        ],
      },
      {
        source: "/api/(.*)",
        headers: [
          { key: "Cache-Control", value: "no-store, max-age=0" },
          { key: "Content-Security-Policy", value: cspValue },
        ],
      },
      {
        source: "/api/og/:path*",
        headers: [
          { key: "Cache-Control", value: "public, max-age=31536000, s-maxage=31536000, immutable" },
          { key: "Content-Security-Policy", value: cspValue },
        ],
      },
      {
        source: "/(.*)",
        headers: [
          { key: "Cache-Control", value: "public, max-age=0, s-maxage=3600, stale-while-revalidate=60" },
          { key: "Content-Security-Policy", value: cspValue },
        ],
      },
    ];
  },
  webpack: (config, { isServer }) => {
    if (isServer) {
      config.externals.push({
        "@prisma/client": "@prisma/client",
      });
    }
    return config;
  },
};

export default nextConfig;
