# Model Deployment & Serving Series — Roadmap

**Series**: "Model Deployment & Serving: From Notebook to Production"
**Folder**: content/blog/machine-learning/model-serving/
**Kit**: .cache/blog-writer/_model-serving-series-kit.md
**Cache**: .cache/blog-writer/_model-serving/<slug>/
**Render**: .cache/blog-writer/_render-serving.sh <slug>
**Total**: 55 posts across 9 tracks (A–I)
**Date**: 2026-06-22

## Progress

### Track A — Foundations (Wave 1)
- [ ] A1 `what-is-model-serving`
- [ ] A2 `the-model-serving-stack`
- [ ] A3 `rest-vs-grpc-vs-streaming-for-ml-apis`
- [ ] A4 `batching-fundamentals-latency-throughput-tradeoff`
- [ ] A5 `model-serving-slas-and-metrics`
- [ ] A6 `model-packaging-and-formats`

### Track B — Inference Runtimes (Wave 2)
- [ ] B1 `torchserve-deep-dive`
- [ ] B2 `triton-inference-server-deep-dive`
- [ ] B3 `onnx-runtime-for-serving`
- [ ] B4 `ray-serve-deep-dive`
- [ ] B5 `bentoml-and-mlserver`
- [ ] B6 `choosing-your-serving-stack`

### Track C — LLM Serving Core (Wave 3)
- [ ] C1 `why-llm-serving-is-different`
- [ ] C2 `continuous-batching-and-pagedattention`
- [ ] C3 `vllm-deep-dive`
- [ ] C4 `text-generation-inference-deep-dive`
- [ ] C5 `streaming-and-sse-for-llms`
- [ ] C6 `quantization-for-llm-serving`
- [ ] C7 `multi-lora-and-adapter-serving`

### Track D — Infrastructure & Scaling (Wave 4)
- [ ] D1 `containerizing-ml-models`
- [ ] D2 `kubernetes-for-ml-serving`
- [ ] D3 `gpu-scheduling-and-mig`
- [ ] D4 `autoscaling-model-servers`
- [ ] D5 `load-balancing-for-inference`
- [ ] D6 `multi-region-and-edge-serving`

### Track E — Optimization (Wave 5)
- [ ] E1 `dynamic-batching-deep-dive`
- [ ] E2 `quantization-for-inference-not-training`
- [ ] E3 `kernel-fusion-and-torch-compile`
- [ ] E4 `kv-cache-optimization`
- [ ] E5 `speculative-decoding-in-production`
- [ ] E6 `tensor-and-pipeline-parallelism-for-serving`
- [ ] E7 `cpu-inference-and-heterogeneous-serving`

### Track F — Reliability & Monitoring (Wave 6)
- [ ] F1 `canary-and-ab-testing-for-models`
- [ ] F2 `shadow-mode-and-champion-challenger`
- [ ] F3 `drift-detection-in-serving`
- [ ] F4 `observability-for-model-servers`
- [ ] F5 `error-handling-and-fallbacks-in-serving`
- [ ] F6 `rate-limiting-and-fairness-for-inference`

### Track G — MLOps & CI/CD (Wave 7)
- [ ] G1 `model-registry-and-versioning`
- [ ] G2 `cicd-for-model-deployments`
- [ ] G3 `feature-stores-for-online-inference`
- [ ] G4 `model-rollback-patterns`
- [ ] G5 `cost-management-for-serving`
- [ ] G6 `serving-governance-and-compliance`

### Track H — Large-Scale LLM Infrastructure (Wave 8)
- [ ] H1 `prefill-decode-disaggregation`
- [ ] H2 `llm-control-planes-aibrix-kserve`
- [ ] H3 `multi-node-llm-serving-100b-plus`
- [ ] H4 `deepseek-inference-optimization`
- [ ] H5 `high-concurrency-slo-management`
- [ ] H6 `hardware-ecosystem-for-llm-serving`
- [ ] H7 `cost-optimization-at-llm-scale`

### Track I — Case Studies + Capstone (Wave 9)
- [ ] I1 `serving-a-vision-model-at-scale`
- [ ] I2 `serving-an-llm-chatbot-end-to-end`
- [ ] I3 `multi-modal-serving`
- [ ] I4 `the-model-serving-playbook` ← CAPSTONE

## Commit log
(fill in as waves ship)
