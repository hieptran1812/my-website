---
title: "Rewritten Kernels for GLM 5.3 Flash Serving"
publishDate: "2026-09-30"
date: "2026-09-30"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "serving", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "A provider says it rewrote GLM 5.3 Flash kernels, reaching 670 tokens per second and running on AMD."
---

> Source: [@rightnow_co on X](https://x.com/rightnow_co/status/2105186912634057086) · posted 2026-09-30

## Summary

The author says they spent September rewriting the kernels behind GLM 5.3 Flash and released it on RunInfra. They claim 670 tokens per second on Vercel AI Gateway, prices of $0.11 per 1M input, $0.45 output and $0.03 cached, 1M token context, FP8, and that it now runs on AMD.

## Background

Custom kernels tuned for a model's specific shapes can greatly speed up serving. FP8 is an 8-bit number format that cuts memory and speeds up math with small accuracy loss.

## What I took away

- Kernel work is a major lever on serving cost.
- Speed numbers are the provider's claims; benchmark yourself.

## Reference

- Original post: [https://x.com/rightnow_co/status/2105186912634057086](https://x.com/rightnow_co/status/2105186912634057086)
- Author: [@rightnow_co](https://x.com/rightnow_co) on X, posted 2026-09-30
