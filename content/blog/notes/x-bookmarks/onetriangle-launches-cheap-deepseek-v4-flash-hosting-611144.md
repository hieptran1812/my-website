---
title: "OneTriangle Launches Cheap DeepSeek V4 Flash Hosting"
publishDate: "2026-08-24"
date: "2026-08-24"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "kv-cache", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "OneTriangle, a YC startup, hosts DeepSeek V4 Flash cheaply using KV cache optimizations."
---

> Source: [@agupta on X](https://x.com/agupta/status/2091954753307611144) · posted 2026-08-24

## Summary

The author praises a team that made several technical advances in KV cache optimization to build a competitive inference service. The quoted post says **OneTriangle** (YC S26) is building fast, cheap inference and hosting **DeepSeek V4 Flash** at $0.15/M input and $0.35/M output tokens, with a team of MIT graduates and olympiad medalists, where the text is cut off.

## Background

The KV cache stores past attention data so the model does not recompute it, and it is a major memory cost in serving. Better KV cache handling lets more requests fit per GPU, which lowers price per token.

## What I took away

- KV cache efficiency is a main lever for cheaper inference.
- New providers compete on price for popular open models.
- Check reliability and rate limits, not just price, before switching providers.

## Reference

- Original post: [https://x.com/agupta/status/2091954753307611144](https://x.com/agupta/status/2091954753307611144)
- Author: [@agupta](https://x.com/agupta) on X, posted 2026-08-24
