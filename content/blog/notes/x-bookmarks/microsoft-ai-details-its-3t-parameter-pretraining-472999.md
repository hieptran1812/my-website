---
title: "Microsoft AI Details Its 3T-Parameter Pretraining"
publishDate: "2026-10-15"
date: "2026-10-15"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "moe", "llm-architecture"]
author: "Hiep Tran"
featured: false
excerpt: "A researcher praises Microsoft AI's technical blog on pretraining a 3T-parameter base model and compares its shape to Kimi K3."
---

> Source: [@eliebakouch on X](https://x.com/eliebakouch/status/2110779696296472999) · posted 2026-10-15

## Summary

@eliebakouch calls Microsoft's technical blog post on its new 3T pretraining amazing, with rare and clear insights on training models this size. He notes the model is sparser than Kimi K3 (80B active vs 104B) and uses the same sliding-window hybrid design. The quoted Microsoft AI post says they built state-of-the-art 3T-parameter base models in-house, with post-training results coming.

## Background

Active parameters are the parameters used per token in a mixture-of-experts model, much smaller than the total. Sparser means a lower active-to-total ratio, saving compute. Sliding-window hybrid attention mixes local attention over nearby tokens with some full attention layers.

## What I took away

- Frontier models are converging on sparse MoE with hybrid attention.
- Detailed training blogs from big labs are rare and worth reading.
- Base model results come before post-training; final quality may differ.

## Reference

- Original post: [https://x.com/eliebakouch/status/2110779696296472999](https://x.com/eliebakouch/status/2110779696296472999)
- Author: [@eliebakouch](https://x.com/eliebakouch) on X, posted 2026-10-15
