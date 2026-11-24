---
title: "Visual Breakdown of a Large MoE Model Design"
publishDate: "2026-07-16"
date: "2026-07-16"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "moe", "llm-architecture", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "A visual companion to a breakdown of a 975B MoE model with 41B active parameters and two shared experts."
---

> Source: [@nrehiew_ on X](https://x.com/nrehiew_/status/2077559125270241490) · posted 2026-07-16

## Summary

The author adds a visual for the quoted analysis. The quoted post describes an MoE model with 975B total and 41B active parameters, trained on 45T tokens of multimodal data, about 10% more active parameters than DeepSeek V3 and 3x the tokens. It notes a DeepSeek V3 style design with 6 routed experts out of 256 and 2 shared experts, which it calls unusual since most use 1 or 0.

## Background

In MoE models, total parameters are all experts combined, while active parameters are what each token uses. Shared experts are always used by every token, handling common knowledge, while routed experts specialize. More shared experts raise the fixed compute per token.

## What I took away

- Compare models by active parameters and token count, not only total size.
- Shared expert count is a small design choice with cost effects.
- Open the link for the diagram.

## Reference

- Original post: [https://x.com/nrehiew_/status/2077559125270241490](https://x.com/nrehiew_/status/2077559125270241490)
- Author: [@nrehiew_](https://x.com/nrehiew_) on X, posted 2026-07-16
