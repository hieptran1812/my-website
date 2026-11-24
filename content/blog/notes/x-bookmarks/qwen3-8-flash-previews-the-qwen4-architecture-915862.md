---
title: "Qwen3.8-Flash Previews the Qwen4 Architecture"
publishDate: "2026-08-26"
date: "2026-08-26"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "open-models", "moe", "llm-architecture"]
author: "Hiep Tran"
featured: false
excerpt: "Qwen3.8-Flash is an open-weight multimodal MoE previewing the Qwen4 architecture, with 125B parameters plus 51B N-gram parameters."
---

> Source: [@xuanmingzhangai on X](https://x.com/xuanmingzhangai/status/2092628670745915862) · posted 2026-08-26

## Summary

The author says Qwen-4 is coming and that the team is first testing the limits of pretraining architecture. The quoted post introduces **Qwen3.8-Flash**, an open-weight multimodal MoE and early preview of the Qwen4 architecture. The production version will be on QwenCloud API at $0.16 per 1M input and $0.47 per 1M output tokens, and the model has 125B parameters plus 51B N-gram parameters.

## Background

MoE (mixture of experts) routes each token to a few expert sub-networks, so only part of the model runs per token. N-gram parameters likely refer to a lookup-style memory keyed on short token sequences, a way to add capacity cheaply. Multimodal means it handles inputs like images as well as text.

## What I took away

- Preview models hint at the architecture of the next major release.
- Extra memory-style parameters are one way labs add capacity without more compute.
- Note the difference between open weights and the hosted production version.

## Reference

- Original post: [https://x.com/xuanmingzhangai/status/2092628670745915862](https://x.com/xuanmingzhangai/status/2092628670745915862)
- Author: [@xuanmingzhangai](https://x.com/xuanmingzhangai) on X, posted 2026-08-26
