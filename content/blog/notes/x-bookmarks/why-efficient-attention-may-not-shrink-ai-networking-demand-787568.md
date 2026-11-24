---
title: "Why Efficient Attention May Not Shrink AI Networking Demand"
publishDate: "2026-07-18"
date: "2026-07-18"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "hardware", "industry-news", "kv-cache"]
author: "Hiep Tran"
featured: false
excerpt: "SemiAnalysis argues that Kimi K3's KDA attention, despite cutting KV-transfer bandwidth, will not shrink the AI networking switch market."
---

> Source: [@SemiAnalysis_ on X](https://x.com/SemiAnalysis_/status/2078594446996787568) · posted 2026-07-18

## Summary

SemiAnalysis compares the moment to DeepSeek in January 2025, when people feared efficiency gains would reduce hardware demand. They say some may think the AI networking switch market will shrink because Kimi K3 uses KDA attention, which reduces KV-transfer bandwidth by up to 10x. They claim the opposite is true and explain why in a thread.

## Background

In large deployments, KV cache data is often moved between machines, for example between prefill and decode servers, which uses network bandwidth. TAM means total addressable market. The general argument that efficiency raises total usage is often called Jevons paradox.

## What I took away

- Efficiency gains in one layer can increase total demand rather than reduce it.
- Treat market predictions in threads as the author's argument and read the full reasoning.
- KV transfer bandwidth is a real cost driver in disaggregated serving.

## Reference

- Original post: [https://x.com/SemiAnalysis_/status/2078594446996787568](https://x.com/SemiAnalysis_/status/2078594446996787568)
- Author: [@SemiAnalysis_](https://x.com/SemiAnalysis_) on X, posted 2026-07-18
