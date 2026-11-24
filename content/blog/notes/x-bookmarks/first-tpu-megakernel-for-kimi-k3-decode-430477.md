---
title: "First TPU Megakernel for Kimi K3 Decode"
publishDate: "2026-09-23"
date: "2026-09-23"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "llm-inference", "speculative-decoding", "hardware"]
author: "Hiep Tran"
featured: false
excerpt: "Inferact reports a single-kernel TPU implementation of Kimi K3 reaching 709 tokens/s on low-concurrency decode."
---

> Source: [@inferact on X](https://x.com/inferact/status/2102824415587430477) · posted 2026-09-23

## Summary

Inferact says its first TPU megakernel for **Kimi K3** reaches 709 tokens/s on low-concurrency decode, versus 450 tokens/s for their GB200 baseline. Both use **DSpark** speculative decoding. They believe it is the first TPU inference megakernel, with the whole model running in a single Pallas kernel.

## Background

A megakernel runs the whole model forward pass inside one GPU or TPU kernel, avoiding the overhead of launching many small kernels. Pallas is the JAX language for writing custom TPU and GPU kernels. Low-concurrency decode, with few users at once, is latency bound, so launch overhead matters a lot there.

## What I took away

- Megakernels target latency when batch sizes are small.
- TPUs can compete with GPUs on decode speed given custom kernels.
- Compare baselines carefully; both runs here used the same speculative decoding.

## Reference

- Original post: [https://x.com/inferact/status/2102824415587430477](https://x.com/inferact/status/2102824415587430477)
- Author: [@inferact](https://x.com/inferact) on X, posted 2026-09-23
