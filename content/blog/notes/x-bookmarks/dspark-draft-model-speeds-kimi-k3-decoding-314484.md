---
title: "DSpark Draft Model Speeds Kimi K3 Decoding"
publishDate: "2026-07-27"
date: "2026-07-27"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "speculative-decoding", "llm-inference", "serving"]
author: "Hiep Tran"
featured: false
excerpt: "RadixArk trained a DSpark speculator for Kimi K3 with SpecForge, raising batch-1 decode from about 113 to about 423 tok/s."
---

> Source: [@radixark on X](https://x.com/radixark/status/2081793554377314484) · posted 2026-07-27

## Summary

RadixArk says it trained a **DSpark** speculator draft model for Kimi K3 using SpecForge, taking batch-1 decode from about 113 to about 423 tok/s. It reports +68% throughput at batch size 256 on chat versus verify-all, lossless, and accept lengths of 5.67 on GSM8K and 5.36 on HumanEval. The quoted post says SGLang ran K3 at 423 tok/s on day 0, with RL support in Miles, using fused KDA decode kernels, DP attention, DSpark and PD disaggregation.

## Background

A speculator drafts tokens that the main model verifies. Lossless means outputs match what the main model would produce alone. PD disaggregation separates prefill (reading the prompt) and decode (generating tokens) onto different GPUs so each runs efficiently.

## What I took away

- Trained drafters can multiply single-request speed for huge models.
- Accept length varies by task; math and code differ.
- Day-0 support needs custom kernels for new architecture parts like KDA.

## Reference

- Original post: [https://x.com/radixark/status/2081793554377314484](https://x.com/radixark/status/2081793554377314484)
- Author: [@radixark](https://x.com/radixark) on X, posted 2026-07-27
