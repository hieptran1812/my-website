---
title: "DPPO And Why PPO Clipping Needs A Better Fix"
publishDate: "2026-06-18"
date: "2026-06-18"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "A post argues DAPO's clip_higher does not address PPO's ratio clipping root cause and points to DPPO."
---

> Source: [@QPHutu on X](https://x.com/QPHutu/status/2067424711848636744) · posted 2026-06-18

## Summary

@QPHutu loves PPO and says DAPO found a problem with ratio clipping, but clip_higher does not fix the root cause. Their DPPO work (arXiv 2602.04879) offers a more principled fix. The quoted post says PPO's second wave comes from the importance-ratio fixing numeric error, async and forward-pass noise biases.

## Background

PPO clips the probability ratio between new and old policies to keep updates small. Clipping affects entropy in unexpected ways. DAPO raised the upper clip, which the author sees as a patch.

## What I took away

- Clipping interacts with entropy.
- Read DPPO for an alternative.

## Reference

- Original post: [https://x.com/QPHutu/status/2067424711848636744](https://x.com/QPHutu/status/2067424711848636744)
- Author: [@QPHutu](https://x.com/QPHutu) on X, posted 2026-06-18
