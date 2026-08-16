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
    // The article page inlines each slug's precomputed ego graph; ship the index
    // so ISR revalidation (which re-runs the server component) can still read it.
    "/blog/[...slug]": ["./src/lib/generated/blogGraph.json"],
  },
  compress: true,
  poweredByHeader: false,
  reactStrictMode: true,
  // The inference-engine / serving-framework posts were consolidated out of
  // model-serving, edge-ai, mlops, large-language-model, ai-agent and
  // open-source-library into content/blog/machine-learning/inference-frameworks/.
  // The slug is the folder path, so every old URL needs a permanent redirect.
  async redirects() {
    return [
      {
        source: "/blog/machine-learning/large-language-model/vllm-inference",
        destination: "/blog/machine-learning/inference-frameworks/vllm-inference",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/large-language-model/sglang-inference",
        destination: "/blog/machine-learning/inference-frameworks/sglang-inference",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/ai-agent/mini-sglang",
        destination: "/blog/machine-learning/inference-frameworks/mini-sglang",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/model-serving/vllm-deep-dive",
        destination: "/blog/machine-learning/inference-frameworks/vllm-deep-dive",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/model-serving/vllm-distributed-architecture-anatomy",
        destination: "/blog/machine-learning/inference-frameworks/vllm-distributed-architecture-anatomy",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/model-serving/debugging-vllm-distributed-serving",
        destination: "/blog/machine-learning/inference-frameworks/debugging-vllm-distributed-serving",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/model-serving/running-vllm-distributed-in-production",
        destination: "/blog/machine-learning/inference-frameworks/running-vllm-distributed-in-production",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/model-serving/text-generation-inference-deep-dive",
        destination: "/blog/machine-learning/inference-frameworks/text-generation-inference-deep-dive",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/model-serving/triton-inference-server-deep-dive",
        destination: "/blog/machine-learning/inference-frameworks/triton-inference-server-deep-dive",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/model-serving/torchserve-deep-dive",
        destination: "/blog/machine-learning/inference-frameworks/torchserve-deep-dive",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/model-serving/ray-serve-deep-dive",
        destination: "/blog/machine-learning/inference-frameworks/ray-serve-deep-dive",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/model-serving/onnx-runtime-for-serving",
        destination: "/blog/machine-learning/inference-frameworks/onnx-runtime-for-serving",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/model-serving/bentoml-and-mlserver",
        destination: "/blog/machine-learning/inference-frameworks/bentoml-and-mlserver",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/model-serving/llm-control-planes-aibrix-kserve",
        destination: "/blog/machine-learning/inference-frameworks/llm-control-planes-aibrix-kserve",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/model-serving/choosing-your-serving-stack",
        destination: "/blog/machine-learning/inference-frameworks/choosing-your-serving-stack",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/model-serving/attention-backends-deep-dive-flashattention-flashinfer",
        destination: "/blog/machine-learning/inference-frameworks/attention-backends-deep-dive-flashattention-flashinfer",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/model-serving/the-model-serving-stack",
        destination: "/blog/machine-learning/inference-frameworks/the-model-serving-stack",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/mlops/tensorrt-end-to-end-inference-compiler",
        destination: "/blog/machine-learning/inference-frameworks/tensorrt-end-to-end-inference-compiler",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/mlops/onnx-deep-dive-format-runtime-serving",
        destination: "/blog/machine-learning/inference-frameworks/onnx-deep-dive-format-runtime-serving",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/edge-ai/running-llms-locally-llama-cpp-and-gguf",
        destination: "/blog/machine-learning/inference-frameworks/running-llms-locally-llama-cpp-and-gguf",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/edge-ai/inference-runtimes-compared",
        destination: "/blog/machine-learning/inference-frameworks/inference-runtimes-compared",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/edge-ai/tensorrt-and-gpu-edge-inference-on-jetson",
        destination: "/blog/machine-learning/inference-frameworks/tensorrt-and-gpu-edge-inference-on-jetson",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/open-source-library/lmcache-kv-cache-layer-deep-dive",
        destination: "/blog/machine-learning/inference-frameworks/lmcache-kv-cache-layer-deep-dive",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/open-source-library/tokenspeed-agentic-inference-engine",
        destination: "/blog/machine-learning/inference-frameworks/tokenspeed-agentic-inference-engine",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/open-source-library/tokenspeed-580-tps-qwen3-5-hybrid-mamba-blackwell",
        destination: "/blog/machine-learning/inference-frameworks/tokenspeed-580-tps-qwen3-5-hybrid-mamba-blackwell",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/edge-ai/running-llms-locally-mlc-and-mobile-stacks",
        destination: "/blog/machine-learning/inference-frameworks/running-llms-locally-mlc-and-mobile-stacks",
        permanent: true,
      },
    ];
  },
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
