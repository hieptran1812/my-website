---
title: "Nemotron Puzzle 75B MoE Release"
publishDate: "2026-07-08"
date: "2026-07-08"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "moe", "open-models", "hardware"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2074669146454044939-1.jpg"
excerpt: "NVIDIA replies to a post about its compressed 75B mixture-of-experts Nemotron model with 1M context."
---

> Source: [@NVIDIAAI on X](https://x.com/NVIDIAAI/status/2074669146454044939) · posted 2026-07-08

![Nemotron Puzzle 75B MoE Release](/imgs/blogs/x-bookmarks/2074669146454044939-1.jpg)

## Summary

@NVIDIAAI replies with a friendly welcome to a post that thanks it for a 75B MoE model. The quoted post lists NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4 with 75.3B total and 9.3B active parameters, compressed from Nemotron-3-Super-120B using the Iterative Puzzle framework, with 1M token context.

## Background

A mixture-of-experts (MoE) model only activates some of its parameters per token, so 9.3B active is much cheaper than 75B total. NVFP4 is a 4-bit floating point format that shrinks memory use. Compression lets large models fit on small machines.

## What I took away

- MoE models can run with a small active compute cost.
- Compressing a bigger model can yield a smaller deployable one.
- Check whether your hardware, such as a single GB10, fits the checkpoint.

## Reference

- Original post: [https://x.com/NVIDIAAI/status/2074669146454044939](https://x.com/NVIDIAAI/status/2074669146454044939)
- Author: [@NVIDIAAI](https://x.com/NVIDIAAI) on X, posted 2026-07-08
