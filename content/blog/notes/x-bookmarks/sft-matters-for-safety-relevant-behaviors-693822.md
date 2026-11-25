---
title: "SFT Matters for Safety-Relevant Behaviors"
publishDate: "2026-06-13"
date: "2026-06-13"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-safety", "post-training", "interpretability"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2065888051738693822-1.jpg"
excerpt: "Neel Nanda says SFT, not just RL, was the source of many Gemini behaviors, which surprised him."
---

> Source: [@NeelNanda5 on X](https://x.com/NeelNanda5/status/2065888051738693822) · posted 2026-06-13

![SFT Matters for Safety-Relevant Behaviors](/imgs/blogs/x-bookmarks/2065888051738693822-1.jpg)

## Summary

@NeelNanda5 says he assumed fixing misalignment mostly meant intervening in RL, and was surprised to be wrong. The quoted GDM interp post says many Gemini behaviors came from the initial supervised finetuning stage rather than later stages like RL.

## Background

SFT (supervised fine-tuning) teaches a model from example answers, while RL optimizes from reward signals. If key behaviors are set in SFT, safety work should look at that data and stage too. The author says results may change over time.

## What I took away

- Do not ignore SFT data when studying misalignment.
- Findings may not hold for future models.

## Reference

- Original post: [https://x.com/NeelNanda5/status/2065888051738693822](https://x.com/NeelNanda5/status/2065888051738693822)
- Author: [@NeelNanda5](https://x.com/NeelNanda5) on X, posted 2026-06-13
