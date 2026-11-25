---
title: "GPT-OSS-20B Implemented From Scratch in PyTorch"
publishDate: "2026-02-03"
date: "2026-02-03"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "moe", "open-models"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2018737452316230084-1.jpg"
excerpt: "The author implemented the GPT-OSS-20B architecture fully from scratch in PyTorch, including RoPE with YaRN and MoE."
---

> Source: [@hamzaelshafie on X](https://x.com/hamzaelshafie/status/2018737452316230084) · posted 2026-02-03

![GPT-OSS-20B Implemented From Scratch in PyTorch (1)](/imgs/blogs/x-bookmarks/2018737452316230084-1.jpg)

![GPT-OSS-20B Implemented From Scratch in PyTorch (2)](/imgs/blogs/x-bookmarks/2018737452316230084-2.jpg)

## Summary

The author says they coded the GPT-OSS-20B architecture from scratch in PyTorch. Components include RoPE with YaRN and NTK-by-parts for context scaling, RMSNorm, SwiGLU with clamping and residuals, Mixture-of-Experts, and optimized self-attention.

## Background

RoPE encodes token positions, and YaRN with NTK-by-parts stretches it to longer contexts. RMSNorm is a normalization layer. Mixture-of-Experts routes each token to a few expert sub-networks to keep compute low.

## What I took away

- Re-implementing a model is a strong way to learn it.
- Long context needs position-encoding tricks.
- Check the code for how MoE routing is written.

## Reference

- Original post: [https://x.com/hamzaelshafie/status/2018737452316230084](https://x.com/hamzaelshafie/status/2018737452316230084)
- Author: [@hamzaelshafie](https://x.com/hamzaelshafie) on X, posted 2026-02-03
