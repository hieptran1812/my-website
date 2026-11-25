---
title: "Argument Against Ollama And Local Alternatives"
publishDate: "2026-03-11"
date: "2026-03-11"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "serving", "open-models"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2031747774786535526-1.jpg"
excerpt: "Ahmad Osman argues against using Ollama, listing alternatives, and quotes a benchmark of Qwen3.5-122B-A10B on several GPU setups."
---

> Source: [@TheAhmadOsman on X](https://x.com/TheAhmadOsman/status/2031747774786535526) · posted 2026-03-11

![Argument Against Ollama And Local Alternatives](/imgs/blogs/x-bookmarks/2031747774786535526-1.jpg)

## Summary

The author says nobody should use ollama, claiming it is slower than llama.cpp on Windows and slower than MLX on Mac, and lists alternatives such as LM Studio, llama.cpp, exllamav2/v3, vLLM, SGLang and TRT-LLM. The quoted post benchmarks Qwen3.5-122B-A10B, reporting 101.4 tok/s on one RTX PRO 6000 (96GB) and 87.0 tok/s on 4x 5090 (128GB).

## Background

Ollama is a popular wrapper for running local models. Tools differ in speed and features. The performance and quality statements here are the author's strong opinion, not neutral measurements.

## What I took away

- Benchmark your own model and hardware before choosing a runtime.
- Treat strong opinions on tools as a starting point.
- One big GPU may beat several smaller ones due to less communication.

## Reference

- Original post: [https://x.com/TheAhmadOsman/status/2031747774786535526](https://x.com/TheAhmadOsman/status/2031747774786535526)
- Author: [@TheAhmadOsman](https://x.com/TheAhmadOsman) on X, posted 2026-03-11
