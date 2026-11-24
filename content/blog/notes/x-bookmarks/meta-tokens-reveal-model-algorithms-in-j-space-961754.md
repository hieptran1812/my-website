---
title: "Meta-Tokens Reveal Model Algorithms in J-Space"
publishDate: "2026-07-21"
date: "2026-07-21"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "interpretability", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "Neel Nanda highlights meta-tokens found with J-Lens, where Chinese characters for what does this mean appear when a model is confused."
---

> Source: [@NeelNanda5 on X](https://x.com/NeelNanda5/status/2079676406053961754) · posted 2026-07-21

## Summary

Neel Nanda says he found **meta-tokens** surprising: when the model is confused about a sentence, Chinese characters meaning what does this mean appear in the J-Space. The quoted post says J-space can surface what a model is thinking and asks whether it can also show how the model computes answers. A new blog post applies J-Lens to find meta-tokens that reveal the algorithm Qwen-3.6-27B uses for a task.

## Background

Interpretability research tries to read what is happening inside a model's internal activations. Lens methods project hidden states onto the vocabulary to see which tokens the model is internally leaning toward. Meta-tokens appear to describe the model's own process rather than the answer.

## What I took away

- Internal states can sometimes be read as words that describe the model's process.
- Multilingual models may think in a different language than the input.
- Treat lens readings as clues, not complete explanations.

## Reference

- Original post: [https://x.com/NeelNanda5/status/2079676406053961754](https://x.com/NeelNanda5/status/2079676406053961754)
- Author: [@NeelNanda5](https://x.com/NeelNanda5) on X, posted 2026-07-21
