---
title: "Techniques For Faster And Cheaper LLM Inference"
publishDate: "2025-11-19"
date: "2025-11-19"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "quantization", "kv-cache", "serving"]
author: "Hiep Tran"
featured: false
excerpt: "A checklist of techniques to master for making LLM inference faster and cheaper, from quantization to batching."
---

> Source: [@athletic_coder on X](https://x.com/athletic_coder/status/1991114674767892539) · posted 2025-11-19

## Summary

The post lists techniques to master for faster and cheaper LLM inference. The visible items are quantization (INT8/INT4/FP8), KV-cache optimization, Flash Attention, speculative decoding, continuous batching, paged attention in the style of vLLM, and tensor (parallelism). The text is cut off after item 7.

## Background

Inference is the step where a trained model generates text for users. Quantization stores weights in fewer bits, the KV cache holds past attention results so they are not recomputed, and continuous batching keeps the GPU busy by mixing requests. Together these cut latency and cost per token.

## What I took away

- Use this list as a study map for inference work.
- Quantization and KV-cache handling both save memory, which often limits throughput.
- Batching and paged memory management matter as much as model-level tricks.
- The post is truncated, so check the link for the full list.

## Reference

- Original post: [https://x.com/athletic_coder/status/1991114674767892539](https://x.com/athletic_coder/status/1991114674767892539)
- Author: [@athletic_coder](https://x.com/athletic_coder) on X, posted 2025-11-19
