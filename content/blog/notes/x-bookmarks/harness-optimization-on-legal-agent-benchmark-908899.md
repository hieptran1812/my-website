---
title: "Harness Optimization On Legal Agent Benchmark"
publishDate: "2026-07-04"
date: "2026-07-04"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "harness-engineering", "ai-agents", "open-models"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2073312966728908899-1.jpg"
excerpt: "A post claims DeepSeek V4 Pro rose from 0 to 5% all-pass on a legal agent benchmark by evolving the harness."
---

> Source: [@joelniklaus on X](https://x.com/joelniklaus/status/2073312966728908899) · posted 2026-07-04

![Harness Optimization On Legal Agent Benchmark](/imgs/blogs/x-bookmarks/2073312966728908899-1.jpg)

## Summary

@joelniklaus claims DeepSeek V4 Pro goes from 0 to 5% all-pass on the Harvey Legal Agent Benchmark, doubling GPT 5.5, through harness optimization. The quoted post summarizes a Hugging Face blog: freeze the model weights and let an automated loop rewrite the code around it.

## Background

A harness is the code around a model: prompts, tools, and control flow. Evolving it can improve results without training. The numbers are the author's claim.

## What I took away

- Improving the harness can matter as much as the model.
- An automated loop can search over harness code.
- Treat reported gains as claims until reproduced.

## Reference

- Original post: [https://x.com/joelniklaus/status/2073312966728908899](https://x.com/joelniklaus/status/2073312966728908899)
- Author: [@joelniklaus](https://x.com/joelniklaus) on X, posted 2026-07-04
