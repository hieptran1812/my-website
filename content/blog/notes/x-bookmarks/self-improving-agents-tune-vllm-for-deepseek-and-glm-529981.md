---
title: "Self-Improving Agents Tune vLLM for DeepSeek and GLM"
publishDate: "2026-07-29"
date: "2026-07-29"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-agents", "llm-inference", "serving"]
author: "Hiep Tran"
featured: false
excerpt: "Asari AI says its self-improving agents optimized vLLM for up to 16% more throughput on DeepSeek v4 Pro and GLM 5.2 on B200s."
---

> Source: [@AsariAILabs on X](https://x.com/AsariAILabs/status/2082517627994529981) · posted 2026-07-29

## Summary

Asari AI says its **self-improving agents** optimized the full vLLM inference stack. It claims up to 16% more throughput and interactivity for DeepSeek v4 Pro and GLM 5.2 on B200 GPUs, without MTP. It says every change was verified and the agents got better and faster with each iteration.

## Background

Interactivity usually means tokens per second seen by one user, while throughput is total tokens across users. MTP (multi-token prediction) is a separate speedup technique that was not used here. Self-improving agents learn from earlier attempts to make later optimizations faster.

## What I took away

- Automated agents can be applied to systems performance work.
- Ask how changes were verified for correctness, not just speed.
- Gains without MTP may stack with MTP later.

## Reference

- Original post: [https://x.com/AsariAILabs/status/2082517627994529981](https://x.com/AsariAILabs/status/2082517627994529981)
- Author: [@AsariAILabs](https://x.com/AsariAILabs) on X, posted 2026-07-29
