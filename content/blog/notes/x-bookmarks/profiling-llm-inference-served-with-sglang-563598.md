---
title: "Profiling LLM Inference Served With SGLang"
publishDate: "2026-08-08"
date: "2026-08-08"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "serving", "gpu-kernels", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "A blog post that profiles LLM inference on SGLang and explains the kernels, patterns and bottlenecks seen in production."
---

> Source: [@jino_rohit on X](https://x.com/jino_rohit/status/2085947942339563598) · posted 2026-08-08

## Summary

The author shares a blog post they call their favorite piece of writing so far. In it they profile LLM inference served with **SGLang** and reason about the patterns, kernels and bottlenecks that usually show up in production.

## Background

SGLang is an open-source engine for serving large language models. Profiling means recording what the GPU spends its time on, kernel by kernel, so you can see whether a workload is limited by compute, memory bandwidth or overhead such as scheduling.

## What I took away

- Profile before optimizing; the trace shows where time really goes.
- Learn to read a GPU trace and map kernels back to model layers.
- Production bottlenecks often differ from what microbenchmarks suggest.

## Reference

- Original post: [https://x.com/jino_rohit/status/2085947942339563598](https://x.com/jino_rohit/status/2085947942339563598)
- Author: [@jino_rohit](https://x.com/jino_rohit) on X, posted 2026-08-08
