---
title: "End-To-End Test-Time Training For Long Context"
publishDate: "2026-01-12"
date: "2026-01-12"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2010807762893349067-1.jpg"
excerpt: "Stanford NLP says long context can come from continual learning at test time on existing architectures."
---

> Source: [@stanfordnlp on X](https://x.com/stanfordnlp/status/2010807762893349067) · posted 2026-01-12

![End-To-End Test-Time Training For Long Context](/imgs/blogs/x-bookmarks/2010807762893349067-1.jpg)

## Summary

Stanford NLP presents TTT-E2E, where LLMs keep training at test time via next-token prediction on the context, compressing it into weights. The quoted post says LLM memory is hard, today's methods are hacks, and next-token prediction is already a good compressor so no radical architecture is needed.

## Background

Instead of storing context in a growing cache, the model updates its weights to remember. This gives fixed memory cost.

## What I took away

- Weights can serve as memory.
- Existing architectures may suffice.

## Reference

- Original post: [https://x.com/stanfordnlp/status/2010807762893349067](https://x.com/stanfordnlp/status/2010807762893349067)
- Author: [@stanfordnlp](https://x.com/stanfordnlp) on X, posted 2026-01-12
