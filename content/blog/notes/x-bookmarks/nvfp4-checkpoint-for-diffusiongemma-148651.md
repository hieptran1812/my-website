---
title: "NVFP4 Checkpoint for DiffusionGemma"
publishDate: "2026-06-10"
date: "2026-06-10"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "quantization", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "A post thanking NVIDIA for an NVFP4 checkpoint of DiffusionGemma, quoting launch details."
---

> Source: [@mr_r0b0t on X](https://x.com/mr_r0b0t/status/2064790088098148651) · posted 2026-06-10

## Summary

@mr_r0b0t is happy about an NVFP4 checkpoint for DiffusionGemma. The quoted NVIDIA post says the model generates 256 tokens in parallel per step, with 150+ TPS on DGX Spark and 1,000+ TPS on one H100, with BF16 and NVFP4 checkpoints.

## Background

NVFP4 is a 4-bit floating point format that cuts memory and speeds inference on supported NVIDIA GPUs. A diffusion language model generates many tokens in parallel rather than one at a time. Speeds are vendor claims.

## What I took away

- 4-bit formats make big models fit smaller GPUs.
- Vendor throughput numbers need own testing.

## Reference

- Original post: [https://x.com/mr_r0b0t/status/2064790088098148651](https://x.com/mr_r0b0t/status/2064790088098148651)
- Author: [@mr_r0b0t](https://x.com/mr_r0b0t) on X, posted 2026-06-10
