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
    "/api/blog/search": [
      "./src/lib/generated/searchIndex.json",
      "./src/lib/generated/searchIndex.bin",
    ],
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
        source: "/blog/paper-reading/ai-interpretability/vec2vec-universal-geometry-embeddings",
        destination: "/blog/paper-reading/ai-interpretability/harnessing-the-universal-geometry-of-embeddings",
        permanent: true,
      },
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
      // Model-family posts (DeepSeek, Qwen, Kimi, GLM, Nemotron, MiniMax, Seed)
      // were consolidated into content/blog/machine-learning/frontier-models/.
      {
        source: "/blog/machine-learning/computer-vision/deepseek-ocr-optical-context-compression",
        destination: "/blog/machine-learning/frontier-models/deepseek-ocr-optical-context-compression",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/computer-vision/deepseek-vl-vl2-dynamic-tiling-moe",
        destination: "/blog/machine-learning/frontier-models/deepseek-vl-vl2-dynamic-tiling-moe",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/deep-learning/deepseek-llm-scaling-laws-methodology",
        destination: "/blog/machine-learning/frontier-models/deepseek-llm-scaling-laws-methodology",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/large-language-model/deepseek-coder-repo-packing-fim-continued-pretraining",
        destination: "/blog/machine-learning/frontier-models/deepseek-coder-repo-packing-fim-continued-pretraining",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/large-language-model/deepseek-moe-lineage-fine-grained-shared-experts",
        destination: "/blog/machine-learning/frontier-models/deepseek-moe-lineage-fine-grained-shared-experts",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/large-language-model/deepseek-prover-synthetic-data-rmaxts",
        destination: "/blog/machine-learning/frontier-models/deepseek-prover-synthetic-data-rmaxts",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/large-language-model/deepseek-v3-fp8-mtp-loss-free-balancing",
        destination: "/blog/machine-learning/frontier-models/deepseek-v3-fp8-mtp-loss-free-balancing",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/large-language-model/self-verifiable-reasoning-deepseekmath-v2",
        destination: "/blog/machine-learning/frontier-models/self-verifiable-reasoning-deepseekmath-v2",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/training-techniques/deepseekmath-data-pipeline-and-grpo-origin",
        destination: "/blog/machine-learning/frontier-models/deepseekmath-data-pipeline-and-grpo-origin",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/mlops/deepseek-open-infra-deepep-deepgemm-flashmla",
        destination: "/blog/machine-learning/frontier-models/deepseek-open-infra-deepep-deepgemm-flashmla",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/large-language-model/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning",
        destination: "/blog/machine-learning/frontier-models/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/large-language-model/deepseek-v4-million-token-context-moe",
        destination: "/blog/machine-learning/frontier-models/deepseek-v4-million-token-context-moe",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/large-language-model/guard-models-qwen3guard-streaming-moderation",
        destination: "/blog/machine-learning/frontier-models/guard-models-qwen3guard-streaming-moderation",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/ai-agent/qwen3-coder-next-technical-report",
        destination: "/blog/machine-learning/frontier-models/qwen3-coder-next-technical-report",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/ai-safety/qwen3guard-technical-report",
        destination: "/blog/machine-learning/frontier-models/qwen3guard-technical-report",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/diffusion-model/qwen-image-technical-report",
        destination: "/blog/machine-learning/frontier-models/qwen-image-technical-report",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/large-language-model/qwen-mt-multilingual-translation",
        destination: "/blog/machine-learning/frontier-models/qwen-mt-multilingual-translation",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/large-language-model/qwen3-next-hybrid-attention-ultra-sparse-moe",
        destination: "/blog/machine-learning/frontier-models/qwen3-next-hybrid-attention-ultra-sparse-moe",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/large-language-model/qwen3-technical-report",
        destination: "/blog/machine-learning/frontier-models/qwen3-technical-report",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/multimodal/qwen2-vl-enhancing-vision-language-models-perception-of-the-world-at-any-resolution",
        destination: "/blog/machine-learning/frontier-models/qwen2-vl-enhancing-vision-language-models-perception-of-the-world-at-any-resolution",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/multimodal/qwen3-omni-technical-report",
        destination: "/blog/machine-learning/frontier-models/qwen3-omni-technical-report",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/speech-processing/qwen3-asr-technical-report",
        destination: "/blog/machine-learning/speech-to-text/qwen3-asr-technical-report",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/ai-agent/kimi-researcher",
        destination: "/blog/machine-learning/frontier-models/kimi-researcher",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/large-language-model/kimi-dev",
        destination: "/blog/machine-learning/frontier-models/kimi-dev",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/large-language-model/kimi-k2-5",
        destination: "/blog/machine-learning/frontier-models/kimi-k2-5",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/large-language-model/kimi-k2-6",
        destination: "/blog/machine-learning/frontier-models/kimi-k2-6",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/large-language-model/kimi-k2-open-agentic-intelligence",
        destination: "/blog/machine-learning/frontier-models/kimi-k2-open-agentic-intelligence",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/large-language-model/kimi-k2-thinking",
        destination: "/blog/machine-learning/frontier-models/kimi-k2-thinking",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/large-language-model/kimi-k2",
        destination: "/blog/machine-learning/frontier-models/kimi-k2",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/large-language-model/kimi-k3",
        destination: "/blog/machine-learning/frontier-models/kimi-k3",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/large-language-model/kimi-linear",
        destination: "/blog/machine-learning/frontier-models/kimi-linear",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/multimodal/kimi-vl",
        destination: "/blog/machine-learning/frontier-models/kimi-vl",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/reinforcement-learning/kimi-k1-5",
        destination: "/blog/machine-learning/frontier-models/kimi-k1-5",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/speech-processing/kimi-audio",
        destination: "/blog/machine-learning/frontier-models/kimi-audio",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/reasoning/kimina-prover",
        destination: "/blog/machine-learning/frontier-models/kimina-prover",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/computer-vision/glm-4-1v-thinking-multimodal-reasoning-rl",
        destination: "/blog/machine-learning/frontier-models/glm-4-1v-thinking-multimodal-reasoning-rl",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/large-language-model/autoregressive-blank-infilling-glm",
        destination: "/blog/machine-learning/frontier-models/autoregressive-blank-infilling-glm",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/large-language-model/chatglm-to-glm-4-all-tools",
        destination: "/blog/machine-learning/frontier-models/chatglm-to-glm-4-all-tools",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/large-language-model/glm-4-5-architecture",
        destination: "/blog/machine-learning/frontier-models/glm-4-5-architecture",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/large-language-model/glm-5-2-architecture-techniques-evals",
        destination: "/blog/machine-learning/frontier-models/glm-5-2-architecture-techniques-evals",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/large-language-model/glm-lineage-frontier-llm-technique",
        destination: "/blog/machine-learning/frontier-models/glm-lineage-frontier-llm-technique",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/large-language-model/how-zhipu-measures-glm-benchmarks",
        destination: "/blog/machine-learning/frontier-models/how-zhipu-measures-glm-benchmarks",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/mlops/engineering-glm-130b",
        destination: "/blog/machine-learning/frontier-models/engineering-glm-130b",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/training-techniques/training-glm-4-5-distillation-agentic-rl",
        destination: "/blog/machine-learning/frontier-models/training-glm-4-5-distillation-agentic-rl",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/computer-vision/glm-ocr-training-recipe",
        destination: "/blog/machine-learning/frontier-models/glm-ocr-training-recipe",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/large-language-model/llama-nemotron-efficient-reasoning-models",
        destination: "/blog/machine-learning/frontier-models/llama-nemotron-efficient-reasoning-models",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/large-language-model/nemotron-4-340b-synthetic-data-alignment",
        destination: "/blog/machine-learning/frontier-models/nemotron-4-340b-synthetic-data-alignment",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/large-language-model/nemotron-h-hybrid-mamba-transformer",
        destination: "/blog/machine-learning/frontier-models/nemotron-h-hybrid-mamba-transformer",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation",
        destination: "/blog/machine-learning/frontier-models/nvidia-minitron-pruning-distillation",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/large-language-model/minimax-01-lightning-attention-hybrid-moe",
        destination: "/blog/machine-learning/frontier-models/minimax-01-lightning-attention-hybrid-moe",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/large-language-model/minimax-m1-cispo-test-time-compute",
        destination: "/blog/machine-learning/frontier-models/minimax-m1-cispo-test-time-compute",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/large-language-model/minimax-m2-full-attention-agentic",
        destination: "/blog/machine-learning/frontier-models/minimax-m2-full-attention-agentic",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/large-language-model/minimax-papers-lightning-attention-cispo",
        destination: "/blog/machine-learning/frontier-models/minimax-papers-lightning-attention-cispo",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/large-language-model/minimax-sparse-attention",
        destination: "/blog/machine-learning/frontier-models/minimax-sparse-attention",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/speech-processing/minimax-speech-speaker-encoder-flow-vae",
        destination: "/blog/machine-learning/text-to-speech/minimax-speech-speaker-encoder-flow-vae",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/large-language-model/bytedance-seed-model-universe-by-use-case",
        destination: "/blog/machine-learning/frontier-models/bytedance-seed-model-universe-by-use-case",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/large-language-model/seed-oss-36b-open-long-context-thinking-budget",
        destination: "/blog/machine-learning/frontier-models/seed-oss-36b-open-long-context-thinking-budget",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/large-language-model/seed-coder-model-centric-code-data-pipeline",
        destination: "/blog/machine-learning/frontier-models/seed-coder-model-centric-code-data-pipeline",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/large-language-model/seed-prover-formal-theorem-proving-lean-imo",
        destination: "/blog/machine-learning/frontier-models/seed-prover-formal-theorem-proving-lean-imo",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/large-language-model/seed1-5-thinking-rl-reasoning-vapo-dapo",
        destination: "/blog/machine-learning/frontier-models/seed1-5-thinking-rl-reasoning-vapo-dapo",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/multimodal/seed1-5-vl-native-resolution-vision-language",
        destination: "/blog/machine-learning/frontier-models/seed1-5-vl-native-resolution-vision-language",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/large-language-model/modern-llm-architectures-qwen-llama-gemma-deepseek",
        destination: "/blog/machine-learning/frontier-models/modern-llm-architectures-qwen-llama-gemma-deepseek",
        permanent: true,
      },
      // TTS and STT posts were consolidated into content/blog/machine-learning/
      // text-to-speech/ and speech-to-text/. Series members stayed in their series.
      {
        source: "/blog/machine-learning/signal-processing/kokoro-82m",
        destination: "/blog/machine-learning/text-to-speech/kokoro-82m",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/signal-processing/orpheus-tts-llm-speech-snac",
        destination: "/blog/machine-learning/text-to-speech/orpheus-tts-llm-speech-snac",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/signal-processing/cosyvoice-v1-v3-tokens-flow-matching-streaming",
        destination: "/blog/machine-learning/text-to-speech/cosyvoice-v1-v3-tokens-flow-matching-streaming",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/signal-processing/distilling-fast-tts-non-verbal",
        destination: "/blog/machine-learning/text-to-speech/distilling-fast-tts-non-verbal",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/signal-processing/hifi-gan",
        destination: "/blog/machine-learning/text-to-speech/hifi-gan",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/deep-learning/training-cosyvoice",
        destination: "/blog/machine-learning/text-to-speech/training-cosyvoice",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/frontier-models/minimax-speech-speaker-encoder-flow-vae",
        destination: "/blog/machine-learning/text-to-speech/minimax-speech-speaker-encoder-flow-vae",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/frontier-models/fish-audio-model-family-timeline",
        destination: "/blog/machine-learning/text-to-speech/fish-audio-model-family-timeline",
        permanent: true,
      },
      {
        source: "/blog/paper-reading/speech-processing/fish-audio-s2-controllable-text-to-speech",
        destination: "/blog/machine-learning/text-to-speech/fish-audio-s2-controllable-text-to-speech",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/signal-processing/whisper-under-the-hood",
        destination: "/blog/machine-learning/speech-to-text/whisper-under-the-hood",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/deep-learning/training-asr-models",
        destination: "/blog/machine-learning/speech-to-text/training-asr-models",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/deep-learning/canary-parakeet-fastconformer-asr",
        destination: "/blog/machine-learning/speech-to-text/canary-parakeet-fastconformer-asr",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/frontier-models/qwen3-asr-technical-report",
        destination: "/blog/machine-learning/speech-to-text/qwen3-asr-technical-report",
        permanent: true,
      },
      {
        source: "/blog/machine-learning/frontier-models/whisper-model-family-timeline",
        destination: "/blog/machine-learning/speech-to-text/whisper-model-family-timeline",
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
