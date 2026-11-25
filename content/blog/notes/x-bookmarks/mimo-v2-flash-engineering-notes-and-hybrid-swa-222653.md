---
title: "MiMo-V2-Flash Engineering Notes and Hybrid SWA"
publishDate: "2025-12-16"
date: "2025-12-16"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "open-models", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "A team member shares engineering choices behind MiMo-V2-Flash, starting with a hybrid sliding-window attention design."
---

> Source: [@_LuoFuli on X](https://x.com/_LuoFuli/status/2001002838953222653) · posted 2025-12-16

## Summary

The author announces MiMo-V2-Flash is live and calls it step 2 on their AGI roadmap. They share notes on choices that mattered, starting with a Hybrid SWA architecture. They say it is simple and, in their internal benchmarks, beat other linear-attention options.

## Background

SWA means sliding window attention, where each token looks only at a recent window instead of the full history. This saves memory and compute for long inputs. Hybrid designs mix such layers with full-attention layers. The comparison results are the author's internal claims.

## What I took away

- Sliding-window attention is a simple way to cut long-context cost.
- Mixing window and full attention layers is a common hybrid pattern.
- Internal benchmark wins should be checked against independent tests.

## Reference

- Original post: [https://x.com/_LuoFuli/status/2001002838953222653](https://x.com/_LuoFuli/status/2001002838953222653)
- Author: [@_LuoFuli](https://x.com/_LuoFuli) on X, posted 2025-12-16
