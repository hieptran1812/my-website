---
title: "150 Tokens per Second With Qwen3.8 on One RTX 5090"
publishDate: "2026-10-20"
date: "2026-10-20"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "quantization", "hardware", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "A Qwen ambassador reports 150 tok/s decoding of Qwen3.8 Flash-Next on a single RTX 5090 using a custom engine and quant."
---

> Source: [@QwenDevs on X](https://x.com/QwenDevs/status/2112527557782630622) · posted 2026-10-20

## Summary

@QwenDevs praises work by ambassador Kyle Hessling. The quoted post reports 150 tokens per second decode for programming with Qwen3.8 Flash-Next on one RTX 5090 with 128 GB DDR5 in a non-ideal 4-stick dual-channel setup, and 130 tok/s median on mixed prompts, crediting Niko1221's Strata engine and the ISTA Lab IQ3_S quant.

## Background

Decode speed is how fast a model generates output tokens. IQ3_S is a roughly 3-bit quantization format that shrinks weights so large models fit in limited memory. Splitting a model between GPU memory and system RAM lets consumer hardware run bigger models.

## What I took away

- Aggressive quantization plus a tuned engine can make large models usable locally.
- System RAM bandwidth matters when part of the model lives off the GPU.
- Numbers are single-stream; multi-user throughput will differ.

## Reference

- Original post: [https://x.com/QwenDevs/status/2112527557782630622](https://x.com/QwenDevs/status/2112527557782630622)
- Author: [@QwenDevs](https://x.com/QwenDevs) on X, posted 2026-10-20
