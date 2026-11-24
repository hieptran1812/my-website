---
title: "INTELLECT Policy Optimization for Async RL Mismatch"
publishDate: "2026-10-19"
date: "2026-10-19"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training"]
author: "Hiep Tran"
featured: false
excerpt: "A joke about a neolab IPO points to INTELLECT Policy Optimization, which masks tokens by probability difference in async RL."
---

> Source: [@xeophon on X](https://x.com/xeophon/status/2112289046710112604) · posted 2026-10-19

## Summary

@xeophon jokes about the first neolab to IPO, playing on the acronym. The quoted post introduces INTELLECT Policy Optimization (IPO): in async RL the trainer and inference engine never agree exactly on token probabilities, and while IcePop masks tokens based on the probability ratio, which misjudges both tails, IPO masks on the probability difference instead.

## Background

In async RL, samples come from an inference engine and training happens separately, so the two compute slightly different probabilities for the same token. Masking means ignoring tokens where the mismatch is too large. A ratio can look huge for tiny probabilities even when the absolute difference is small, which is why using the difference may be more sensible.

## What I took away

- Ratio-based filters overreact on rare tokens; absolute differences behave differently.
- Trainer-inference mismatch is a recurring theme in RL stability work.
- Compare with the StableReinforce note in this batch.

## Reference

- Original post: [https://x.com/xeophon/status/2112289046710112604](https://x.com/xeophon/status/2112289046710112604)
- Author: [@xeophon](https://x.com/xeophon) on X, posted 2026-10-19
