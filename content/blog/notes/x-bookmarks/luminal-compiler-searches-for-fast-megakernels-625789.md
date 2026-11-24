---
title: "Luminal Compiler Searches for Fast Megakernels"
publishDate: "2026-08-05"
date: "2026-08-05"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "llm-inference", "tools"]
author: "Hiep Tran"
featured: false
excerpt: "Luminal says its compiler searches a large space of possible megakernels to maximize overlap and bandwidth and FLOP utilization."
---

> Source: [@joefioti on X](https://x.com/joefioti/status/2085014897545625789) · posted 2026-08-05

## Summary

The author says Luminal's compiler is finding very fast megakernels. It searches a large space of possible megakernels and picks ones that push overlap and bandwidth and FLOP utilization as far as possible. He notes the results use no extra quantization or speculative decoding.

## Background

A megakernel fuses many operations, or even a whole model forward pass, into one GPU kernel, avoiding launch overhead and idle gaps. Search-based compilers try many candidate schedules and keep the fastest, rather than relying on hand-written kernels.

## What I took away

- Fusing work into one kernel removes launch overhead and gaps between ops.
- Automated search can find kernels humans would not write by hand.
- Note when speedups exclude tricks like quantization; it makes comparison fairer.

## Reference

- Original post: [https://x.com/joefioti/status/2085014897545625789](https://x.com/joefioti/status/2085014897545625789)
- Author: [@joefioti](https://x.com/joefioti) on X, posted 2026-08-05
