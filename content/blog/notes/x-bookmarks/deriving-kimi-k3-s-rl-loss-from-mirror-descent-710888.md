---
title: "Deriving Kimi K3's RL Loss From Mirror Descent"
publishDate: "2026-07-30"
date: "2026-07-30"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training", "math"]
author: "Hiep Tran"
featured: false
excerpt: "The author shows Kimi K3's RL loss follows cleanly from a mirror descent formulation with a softmax policy."
---

> Source: [@manan_tomar on X](https://x.com/manan_tomar/status/2082890412356710888) · posted 2026-07-30

## Summary

The author says **Kimi K3's RL loss** has a clean derivation. Starting from a Mirror Descent formulation and using a softmax form for the policy, which is how token probabilities are computed, it leads to a symmetric loss. That loss uses clipping, among other parts (the text is cut off).

## Background

Mirror descent is an optimization method that takes steps while staying close to the previous solution under some distance measure, often KL divergence. In RL for LLMs, the policy is the model's distribution over tokens, produced by softmax. Clipping limits how much one update can change the policy, which keeps training stable.

## What I took away

- Many RL loss tricks can be derived from first principles rather than guessed.
- Understanding the derivation helps judge when a loss will be stable.
- Read the full thread for the symmetric loss details.

## Reference

- Original post: [https://x.com/manan_tomar/status/2082890412356710888](https://x.com/manan_tomar/status/2082890412356710888)
- Author: [@manan_tomar](https://x.com/manan_tomar) on X, posted 2026-07-30
