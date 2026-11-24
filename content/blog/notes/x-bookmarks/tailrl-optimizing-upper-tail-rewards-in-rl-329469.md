---
title: "TailRL: Optimizing Upper-Tail Rewards in RL"
publishDate: "2026-09-06"
date: "2026-09-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "TailRL extends maximum-likelihood RL to continuous rewards by maximizing the expected log of upper-tail probabilities."
---

> Source: [@rsalakhu on X](https://x.com/rsalakhu/status/2096727730209329469) · posted 2026-09-06

## Summary

The quoted post asks whether RL optimizes the right objective, noting standard RL pulls on the mean and often collapses the output distribution, killing the tail. It introduces **Tail-Likelihood Reinforcement Learning (TailRL)**. The author adds that TailRL extends maximum-likelihood RL from binary to continuous rewards by maximizing the expected log of upper-tail probabilities instead of only mean reward.

## Background

Most RL for LLMs maximizes average reward, which can make the model give the same answer every time. Keeping a diverse tail of good answers helps best-of-k sampling and further training. Continuous rewards are scores rather than simple pass or fail.

## What I took away

- Maximizing mean reward can reduce output diversity.
- Choose the RL objective to match how you will sample at test time.
- Compare with TailSFT, a related idea on the SFT side.

## Reference

- Original post: [https://x.com/rsalakhu/status/2096727730209329469](https://x.com/rsalakhu/status/2096727730209329469)
- Author: [@rsalakhu](https://x.com/rsalakhu) on X, posted 2026-09-06
