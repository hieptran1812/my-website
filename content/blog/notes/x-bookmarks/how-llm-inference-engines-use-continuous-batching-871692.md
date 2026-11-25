---
title: "How LLM Inference Engines Use Continuous Batching"
publishDate: "2025-11-26"
date: "2025-11-26"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "serving", "kv-cache"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1993729773705871692-1.jpg"
excerpt: "A Hugging Face blog post explains continuous batching, KV caching, masking and chunked prefill in inference engines like vLLM."
---

> Source: [@NielsRogge on X](https://x.com/NielsRogge/status/1993729773705871692) · posted 2025-11-26

![How LLM Inference Engines Use Continuous Batching](/imgs/blogs/x-bookmarks/1993729773705871692-1.jpg)

## Summary

The author says an amazing Hugging Face blog post explains how LLM inference engines like vLLM work. It covers continuous batching, KV-caching, attention masking, chunked prefill and decoding. The explanation of continuous batching is cut off.

## Background

Continuous batching lets a server add new requests to a running batch as others finish, instead of waiting for a whole batch to end. Chunked prefill splits a long prompt into pieces so decoding of other requests is not blocked. These techniques raise throughput on GPUs.

## What I took away

- Continuous batching keeps GPUs busy by swapping requests in and out.
- Prefill (reading the prompt) and decode (generating) have different costs.
- A good starting point for learning how serving works.

## Reference

- Original post: [https://x.com/NielsRogge/status/1993729773705871692](https://x.com/NielsRogge/status/1993729773705871692)
- Author: [@NielsRogge](https://x.com/NielsRogge) on X, posted 2025-11-26
