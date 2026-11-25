---
title: "Self-Speculation With Diffusion Drafting in SGLang"
publishDate: "2026-05-29"
date: "2026-05-29"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "speculative-decoding", "llm-inference", "kv-cache", "diffusion"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2060245957254824246-1.jpg"
excerpt: "Pavlo Molchanov claims a 6.75x speedup using one model that drafts in diffusion mode and verifies in autoregressive mode."
---

> Source: [@PavloMolchanov on X](https://x.com/PavloMolchanov/status/2060245957254824246) · posted 2026-05-29

![Self-Speculation With Diffusion Drafting in SGLang](/imgs/blogs/x-bookmarks/2060245957254824246-1.jpg)

## Summary

The author claims **self-speculation brings a 6.75x real speedup** for generation with SGLang. The same model drafts future tokens in diffusion mode and then verifies them in autoregressive (causal) mode. It uses one model and one KV cache, only changing attention masks.

## Background

Speculative decoding means a draft guesses several tokens and the main model checks them in one pass. Here the draft and the checker are the same model, so no extra model or cache is needed. The speedup figure is the author's claim.

## What I took away

- One model can both draft and verify by switching attention masks.
- Sharing the KV cache removes a common cost of speculative decoding.
- Confirm the speedup conditions (batch size, task) before relying on it.

## Reference

- Original post: [https://x.com/PavloMolchanov/status/2060245957254824246](https://x.com/PavloMolchanov/status/2060245957254824246)
- Author: [@PavloMolchanov](https://x.com/PavloMolchanov) on X, posted 2026-05-29
