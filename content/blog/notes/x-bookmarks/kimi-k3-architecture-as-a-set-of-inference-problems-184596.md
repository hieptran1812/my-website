---
title: "Kimi K3 Architecture as a Set of Inference Problems"
publishDate: "2026-08-04"
date: "2026-08-04"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "serving", "moe", "kv-cache"]
author: "Hiep Tran"
featured: false
excerpt: "vLLM recommends a SemiAnalysis writeup on Kimi K3, framing each architecture idea as an inference problem in agentic traces."
---

> Source: [@vllm_project on X](https://x.com/vllm_project/status/2084590702009184596) · posted 2026-08-04

## Summary

vLLM says that in agentic traces each **Kimi K3** architecture idea becomes an inference problem: prefix caching over state rather than tokens, a persistent residual stream, and an MoE path bound by all-to-all communication. They recommend a SemiAnalysis writeup and the vLLM day-0 post it cites. The quoted SemiAnalysis post covers Kimi K3's compressed memory, attention across depth, latent expert routing and serving performance.

## Background

Prefix caching normally saves KV cache for repeated prompt tokens; if a model keeps compressed state instead, the cache must store that state. All-to-all is the communication step where each GPU sends tokens to the GPUs holding their experts, often the MoE bottleneck.

## What I took away

- New architectures change what serving engines must cache and move.
- In MoE serving, communication can matter more than compute.
- Read model architecture and serving writeups together.

## Reference

- Original post: [https://x.com/vllm_project/status/2084590702009184596](https://x.com/vllm_project/status/2084590702009184596)
- Author: [@vllm_project](https://x.com/vllm_project) on X, posted 2026-08-04
