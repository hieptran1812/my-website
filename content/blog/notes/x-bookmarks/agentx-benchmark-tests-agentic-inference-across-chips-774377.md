---
title: "AgentX Benchmark Tests Agentic Inference Across Chips"
publishDate: "2026-08-25"
date: "2026-08-25"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "hardware", "kv-cache"]
author: "Hiep Tran"
featured: false
excerpt: "vLLM shares SemiAnalysis AgentX 1.0, an open agentic coding benchmark from real traces testing GB300, MI355 and B200."
---

> Source: [@vllm_project on X](https://x.com/vllm_project/status/2092040745842774377) · posted 2026-08-25

## Summary

vLLM congratulates SemiAnalysis on **AgentX 1.0**, an open-source multi-turn agentic coding benchmark from about $3M of real traces, running on 1000+ chips and about 2MW of compute, and says vLLM performed competitively. The quoted SemiAnalysis post asks whether the CUDA moat holds up in agentic inference and lists an open $3M dataset, 1M+ context, multi-turn with sub-agents, 95%+ KV cache hit rate, and GB300 NVL72, MI355 and B200 hardware.

## Background

The CUDA moat is the idea that NVIDIA's software ecosystem keeps it ahead of rivals like AMD. Agentic inference has very long, repeated contexts, so KV cache hit rate is high and memory handling matters most. Testing across chips shows how hardware and software compare on that pattern.

## What I took away

- High KV cache hit rates are typical of agent workloads.
- Cross-vendor benchmarks on real traces help test claims about NVIDIA's lead.
- Read the full report for per-chip results rather than headlines.

## Reference

- Original post: [https://x.com/vllm_project/status/2092040745842774377](https://x.com/vllm_project/status/2092040745842774377)
- Author: [@vllm_project](https://x.com/vllm_project) on X, posted 2026-08-25
