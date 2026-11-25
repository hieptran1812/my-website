---
title: "MiniMax Early Int4 QAT Experience"
publishDate: "2025-12-27"
date: "2025-12-27"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "quantization", "open-models", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "MiniMax's Skyler Miao says the team had near-lossless int4 PTQ and QAT since late 2023."
---

> Source: [@SkylerMiao7 on X](https://x.com/SkylerMiao7/status/2004887153290117599) · posted 2025-12-27

## Summary

Skyler Miao answers a question about 4 bit QAT, saying that as early as abab5.5 in late 2023 they achieved near-lossless int4 PTQ and int4 QAT. He says their QAT generalizes well, can move to new models with almost no extra effort, and stayed in use through MiniMax M1. The quoted post asks whether the team looked into 4 bit QAT.

## Background

PTQ (post-training quantization) shrinks weights after training. QAT (quantization-aware training) trains the model while simulating low precision so it loses less accuracy. Int4 means 4-bit integers, which cut memory a lot. Near-lossless is the author's claim.

## What I took away

- QAT can keep accuracy at 4 bits better than PTQ alone.
- A reusable QAT pipeline can carry over between model versions.

## Reference

- Original post: [https://x.com/SkylerMiao7/status/2004887153290117599](https://x.com/SkylerMiao7/status/2004887153290117599)
- Author: [@SkylerMiao7](https://x.com/SkylerMiao7) on X, posted 2025-12-27
