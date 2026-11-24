---
title: "Cursor Open-Sources Mixture-of-Kittens MoE Kernels"
publishDate: "2026-08-04"
date: "2026-08-04"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "moe", "gpu-kernels", "hardware"]
author: "Hiep Tran"
featured: false
excerpt: "Elie Bakouch highlights Cursor's open-source MoE training megakernel for NVL72, noting up to about 2x forward speedups and MXFP8 use."
---

> Source: [@eliebakouch on X](https://x.com/eliebakouch/status/2084673379563421877) · posted 2026-08-04

## Summary

Elie Bakouch says Cursor's open-source Blackwell and NVL72 kernels for MoE look very good, with up to about 2x speedup on the forward pass plus backward support. He notes they use MXFP8 and not NVFP4. The quoted Cursor post introduces **Mixture-of-Kittens (MoK)**, a training megakernel that fuses all MoE communication and computation into one deterministic kernel, claiming up to 2.37x over the strongest public baselines.

## Background

Mixture-of-Experts (MoE) models route each token to a few expert sub-networks, which requires heavy communication between GPUs. NVL72 is an NVIDIA rack of 72 Blackwell GPUs linked by NVLink. MXFP8 and NVFP4 are low-precision number formats; deterministic means the same inputs always give exactly the same outputs.

## What I took away

- Fusing communication with compute is a key lever for MoE training speed.
- Determinism helps debugging and reproducible training runs.
- The choice of MXFP8 over NVFP4 hints at precision trade-offs in training.

## Reference

- Original post: [https://x.com/eliebakouch/status/2084673379563421877](https://x.com/eliebakouch/status/2084673379563421877)
- Author: [@eliebakouch](https://x.com/eliebakouch) on X, posted 2026-08-04
