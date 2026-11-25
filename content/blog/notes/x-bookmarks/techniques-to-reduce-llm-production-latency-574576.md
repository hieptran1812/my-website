---
title: "Techniques To Reduce LLM Production Latency"
publishDate: "2025-12-17"
date: "2025-12-17"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "serving", "speculative-decoding", "kv-cache"]
author: "Hiep Tran"
featured: false
excerpt: "athletic_coder lists topics to master for faster LLM latency, including prefill versus decode, prefix caching and speculative decoding."
---

> Source: [@athletic_coder on X](https://x.com/athletic_coder/status/2001268733218574576) · posted 2025-12-17

## Summary

The author lists techniques to boost LLM latency in production: separating first-token latency from throughput, optimizing prefill versus decode, KV-cache warmup, prefix caching for system prompts, chunked prefill and speculative decoding, plus more not shown.

## Background

Prefill processes the whole prompt and is compute-heavy, while decode generates one token at a time and is memory-bound. Prefix caching reuses work on a shared prompt start, chunked prefill breaks long prompts into pieces so decoding is not blocked, and speculative decoding drafts tokens for a big model to verify.

## What I took away

- Measure time-to-first-token and throughput separately.
- Prefill and decode have different bottlenecks.
- Use the list as a checklist for tuning an inference stack.

## Reference

- Original post: [https://x.com/athletic_coder/status/2001268733218574576](https://x.com/athletic_coder/status/2001268733218574576)
- Author: [@athletic_coder](https://x.com/athletic_coder) on X, posted 2025-12-17
