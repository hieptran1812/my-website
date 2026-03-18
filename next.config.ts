import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  trailingSlash: true,
  images: {
    formats: ["image/avif", "image/webp"],
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048],
    imageSizes: [16, 32, 48, 64, 96, 128, 256],
    minimumCacheTTL: 31536000, // Cache for 1 year
    dangerouslyAllowSVG: true,
    contentDispositionType: "attachment",
    contentSecurityPolicy: "default-src 'self'; script-src 'none'; sandbox;",
  },
  experimental: {
    optimizePackageImports: ["react-icons", "d3", "katex", "gray-matter"],
    scrollRestoration: true,
  },
  compress: true,
  poweredByHeader: false,
  reactStrictMode: true,
  async headers() {
    const cspValue =
      "connect-src 'self' https://vitals.vercel-insights.com https://vercel-insights.com; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://vercel.live;";
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
        source: "/(.*)",
        headers: [
          { key: "Cache-Control", value: "public, max-age=0, s-maxage=3600, stale-while-revalidate=60" },
          { key: "Content-Security-Policy", value: cspValue },
        ],
      },
    ];
  },
  // Webpack configuration for Prisma
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
