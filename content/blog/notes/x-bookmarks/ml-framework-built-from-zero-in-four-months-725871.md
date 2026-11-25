---
title: "ML Framework Built From Zero In Four Months"
publishDate: "2026-04-18"
date: "2026-04-18"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "pretraining", "learning-resources"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2045316682445725871-1.jpg"
excerpt: "NVIDIA AI Dev praises a builder who made a ML framework with Rust and CUDA kernels in four months."
---

> Source: [@NVIDIAAIDev on X](https://x.com/NVIDIAAIDev/status/2045316682445725871) · posted 2026-04-18

![ML Framework Built From Zero In Four Months](/imgs/blogs/x-bookmarks/2045316682445725871-1.jpg)

## Summary

The post is amazed that someone went from knowing zero ML to building an ML framework in 4 months. The quoted builder trained a 12M parameter LLM on a framework with a Rust backend and custom CUDA kernels for flash attention, AdamW and more. They also wrote the transformer and a BPE tokenizer.

## Background

Flash attention is a fast, memory-efficient way to compute attention on GPUs. AdamW is a common optimizer. BPE is a way to split text into tokens. Building all of these teaches how LLM training works.

## What I took away

- Building from scratch is a strong way to learn.
- Small models like 12M parameters are enough to learn the stack.
- Custom kernels teach where the cost lives.

## Reference

- Original post: [https://x.com/NVIDIAAIDev/status/2045316682445725871](https://x.com/NVIDIAAIDev/status/2045316682445725871)
- Author: [@NVIDIAAIDev](https://x.com/NVIDIAAIDev) on X, posted 2026-04-18
