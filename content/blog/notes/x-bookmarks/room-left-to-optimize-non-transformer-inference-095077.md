---
title: "Room Left to Optimize Non-Transformer Inference"
publishDate: "2026-10-12"
date: "2026-10-12"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "llm-architecture", "speculative-decoding"]
author: "Hiep Tran"
featured: false
excerpt: "Albert Gu argues alternative architectures still have many untapped inference optimizations, citing ReplaySSM shipping in production engines."
---

> Source: [@_albertgu on X](https://x.com/_albertgu/status/2109744837432095077) · posted 2026-10-12

## Summary

The author says Transformer inference benefited from dozens of tricks built by the whole ecosystem, and that alternative architectures still have many ideas waiting to be found. The quoted post announces that **ReplaySSM** runs in production in GLM-5.3-Flash and Kimi-K3 and ships in TensorRT-LLM, vLLM, FlashInfer and SGLang. It claims up to 1.47x faster standard decoding, up to 2.1x faster speculative decoding, and support for a low-precision SSM state.

## Background

A state space model (SSM), such as Mamba, is an alternative to attention that keeps a fixed-size running state instead of a growing KV cache. That gives good asymptotic cost for long sequences, but many inference tricks built for Transformers, like speculative decoding and low-precision caches, did not carry over easily. Speculative decoding means a small drafter proposes several tokens and the big model checks them in one pass.

## What I took away

- Architecture comparisons can be unfair if one side has years of inference engineering and the other does not.
- ReplaySSM is a name to look up if you serve hybrid or SSM models.
- Speedup numbers are the authors' claims; check them on your own workload.

## Reference

- Original post: [https://x.com/_albertgu/status/2109744837432095077](https://x.com/_albertgu/status/2109744837432095077)
- Author: [@_albertgu](https://x.com/_albertgu) on X, posted 2026-10-12
