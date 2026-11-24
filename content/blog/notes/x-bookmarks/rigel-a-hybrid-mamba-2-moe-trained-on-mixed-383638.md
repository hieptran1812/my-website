---
title: "Rigel: A Hybrid Mamba-2 MoE Trained on Mixed Hardware"
publishDate: "2026-09-22"
date: "2026-09-22"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "moe", "llm-architecture", "hardware"]
author: "Hiep Tran"
featured: false
excerpt: "Rigel, a 2.3B MoE hybrid Mamba-2 model, reportedly nears Llama-3.2-3B using under 1% of its pretraining FLOPs on mixed hardware."
---

> Source: [@MayankMish98 on X](https://x.com/MayankMish98/status/2102504657973383638) · posted 2026-09-22

## Summary

The author introduces **Rigel**, a 2.3B MoE (360M active) hybrid Mamba-2 model. They claim it lands within a few points of Llama-3.2-3B using under 1% of its pretraining FLOPs. There was no dedicated cluster; the run moved between H100s, A100s, V100s and TPU v5p/v6e on one codebase.

## Background

Mamba-2 is a state space model layer that processes sequences in linear time, often mixed with attention in hybrid models. MoE means only some experts run per token, so active parameters are far fewer than total. Running one training job across many hardware types is hard because each has different speed and features.

## What I took away

- Small teams can pretrain useful models by combining efficient architectures and scattered compute.
- A portable codebase across GPUs and TPUs is a real advantage when compute is scarce.
- Compare models by active parameters and training FLOPs, not only total size.

## Reference

- Original post: [https://x.com/MayankMish98/status/2102504657973383638](https://x.com/MayankMish98/status/2102504657973383638)
- Author: [@MayankMish98](https://x.com/MayankMish98) on X, posted 2026-09-22
