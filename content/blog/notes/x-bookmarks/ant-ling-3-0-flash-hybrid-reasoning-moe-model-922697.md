---
title: "Ant Ling-3.0-flash Hybrid Reasoning MoE Model"
publishDate: "2026-07-23"
date: "2026-07-23"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "open-models", "moe", "llm-architecture"]
author: "Hiep Tran"
featured: false
excerpt: "Teortaxes calls Ant Ling's new Ling-3.0-flash a Kimi K3-lite, a 124B MoE with 5.1B active parameters."
---

> Source: [@teortaxesTex on X](https://x.com/teortaxesTex/status/2080361461616922697) · posted 2026-07-23

## Summary

The author calls it an impressive move from Ant Ling and describes the model as roughly a Kimi K3-lite not made by Moonshot. He says it is about on par with V4-Flash while being more than 2x smaller and faster, using KDA plus MLA and 1/64 sparsity. The quoted post introduces **Ling-3.0-flash**, a hybrid-reasoning MoE for production agents with 124B parameters and 5.1B active, claimed to match or beat their 1T flagship on most shown benchmarks.

## Background

MLA (multi-head latent attention) compresses the KV cache, and KDA is a linear attention variant; mixing them is a hybrid attention design. 1/64 sparsity means only a small fraction of experts are active per token. Hybrid reasoning means the model can either think step by step or answer directly.

## What I took away

- Architecture ideas spread fast across labs after a strong release.
- Smaller active parameter counts mean cheaper serving for agents.
- Benchmark claims versus a flagship should be checked on your tasks.

## Reference

- Original post: [https://x.com/teortaxesTex/status/2080361461616922697](https://x.com/teortaxesTex/status/2080361461616922697)
- Author: [@teortaxesTex](https://x.com/teortaxesTex) on X, posted 2026-07-23
