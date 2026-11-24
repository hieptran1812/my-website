---
title: "Project Ideas for AI Infrastructure Engineers"
publishDate: "2026-10-11"
date: "2026-10-11"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "career", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "A list of hands-on projects for AI infrastructure engineers, starting with tuning vLLM or SGLang serving configs."
---

> Source: [@suraj_sharma14 on X](https://x.com/suraj_sharma14/status/2109252678387724559) · posted 2026-10-11

## Summary

The author lists projects an AI infrastructure engineer should build. The first is tuning a high-throughput serving engine like vLLM or SGLang, adjusting chunked prefill, continuous batching and max-num-seqs for a specific GPU; the author claims default configs waste 60% of VRAM. The quoted post is a learning roadmap that starts with C++, Rust, the CUDA memory hierarchy, PCIe vs NVLink, and writing a custom matrix multiplication kernel.

## Background

Continuous batching lets a server add and remove requests from a running batch instead of waiting for a full batch. Chunked prefill splits long prompts into pieces so they do not block decoding for other users. max-num-seqs caps how many sequences run at once.

## What I took away

- Benchmark serving configs on your own GPU instead of trusting defaults.
- The 60% VRAM waste figure is the author's claim, not a measured fact.
- Learning-by-building beats reading lists for infra skills.

## Reference

- Original post: [https://x.com/suraj_sharma14/status/2109252678387724559](https://x.com/suraj_sharma14/status/2109252678387724559)
- Author: [@suraj_sharma14](https://x.com/suraj_sharma14) on X, posted 2026-10-11
