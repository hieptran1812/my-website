---
title: "FreeToken Claims Faster Local Inference Than Ollama"
publishDate: "2026-08-21"
date: "2026-08-21"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "kv-cache", "tools"]
author: "Hiep Tran"
featured: false
excerpt: "FreeToken claims 3 to 4 times faster decode and 6 to 30 times faster prefill than Ollama via CPU-GPU execution and caching."
---

> Source: [@Andy_ShuoYang on X](https://x.com/Andy_ShuoYang/status/2090856978428145761) · posted 2026-08-21

## Summary

The authors claim **FreeToken** has 3 to 4x faster decode and 6 to 30x faster prefill than Ollama. They credit bandwidth-adaptive CPU-GPU execution and semantic-aware caching across agent turns, with details in an arXiv technical report.

## Background

Prefill is processing the input prompt; decode is generating output tokens one at a time. Bandwidth-adaptive CPU-GPU execution splits work between processors based on memory speed. Caching across agent turns reuses computation when conversations repeat context.

## What I took away

- Prefill and decode have different bottlenecks and should be measured separately.
- Caching across agent turns can greatly speed up local agents.
- The speedups are the authors' claims; another bookmark notes they need verification.

## Reference

- Original post: [https://x.com/Andy_ShuoYang/status/2090856978428145761](https://x.com/Andy_ShuoYang/status/2090856978428145761)
- Author: [@Andy_ShuoYang](https://x.com/Andy_ShuoYang) on X, posted 2026-08-21
