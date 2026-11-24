---
title: "NVIDIA Rubin Architecture Headline Specs"
publishDate: "2026-07-21"
date: "2026-07-21"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "hardware", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "An NVIDIA engineer shares Rubin's headline specs, including 50 PFLOPS sparse NVFP4, 288GB HBM4 and NVLink 6."
---

> Source: [@KranenKyle on X](https://x.com/KranenKyle/status/2079608215957090621) · posted 2026-07-21

## Summary

The author says NVIDIA is revealing **Rubin**, designed to maximize agentic tokens per watt. Headline numbers are 50 petaflops of sparse NVFP4 compute, 288 GB of HBM4 with 22 TB/s bandwidth, and NVLink 6 with 3.6 TB/s. The thread goes on to list new performance features.

## Background

Rubin is NVIDIA's GPU generation after Blackwell. HBM4 is the newest high-bandwidth memory, and memory bandwidth often limits LLM decoding speed. NVLink connects GPUs so they can share work quickly, and NVFP4 is a 4-bit number format.

## What I took away

- Memory bandwidth and capacity matter as much as FLOPS for inference.
- Sparse FLOPS figures assume structured sparsity; dense numbers are lower.
- Tokens per watt is becoming a main metric for AI hardware.

## Reference

- Original post: [https://x.com/KranenKyle/status/2079608215957090621](https://x.com/KranenKyle/status/2079608215957090621)
- Author: [@KranenKyle](https://x.com/KranenKyle) on X, posted 2026-07-21
