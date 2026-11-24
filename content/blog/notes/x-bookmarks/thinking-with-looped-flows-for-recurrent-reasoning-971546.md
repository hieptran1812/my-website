---
title: "Thinking With Looped Flows for Recurrent Reasoning"
publishDate: "2026-09-11"
date: "2026-09-11"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-research", "diffusion", "llm-architecture"]
author: "Hiep Tran"
featured: false
excerpt: "A new approach trains recurrent reasoning with local denoising objectives, claimed SOTA on ARC-AGI-1 and 2 among looped models."
---

> Source: [@ayhozade on X](https://x.com/ayhozade/status/2098391682353971546) · posted 2026-09-11

## Summary

The author asks whether neural networks can learn to think by denoising and introduces **Thinking with Looped Flows**. It trains recurrent reasoning using local denoising objectives. They claim SOTA on ARC-AGI-1 and 2 among looped models.

## Background

Looped models reuse the same layers many times, so they can think longer without more parameters. Denoising, as in diffusion or flow models, means learning to refine a noisy guess step by step. ARC-AGI is a benchmark of abstract visual puzzles meant to test reasoning.

## What I took away

- Iterative refinement is an alternative to token-by-token reasoning.
- Local per-step objectives can make training recurrent models easier.
- Note the SOTA claim is limited to looped models.

## Reference

- Original post: [https://x.com/ayhozade/status/2098391682353971546](https://x.com/ayhozade/status/2098391682353971546)
- Author: [@ayhozade](https://x.com/ayhozade) on X, posted 2026-09-11
