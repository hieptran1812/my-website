---
title: "TailSFT: Preparing Models Better for RL"
publishDate: "2026-09-12"
date: "2026-09-12"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "post-training", "reinforcement-learning", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "A summary of TailSFT, which skips the most-improved examples during SFT to give RL a better pass@k starting point."
---

> Source: [@giffmana on X](https://x.com/giffmana/status/2098681993156255952) · posted 2026-09-12

## Summary

The quoted authors say RL is expensive, so SFT should prepare models well for it, and propose **TailSFT**. The author gives a TL;DR: record the pre-SFT loss of each example, exclude the examples that improved most relative to that loss from backprop during SFT, and get a better pass@k starting point for RL. They find it simple and intuitive.

## Background

SFT (supervised fine-tuning) trains a model on example answers before RL. Pass@k is the chance at least one of k samples is correct; higher pass@k gives RL more successes to learn from. Standard cross-entropy SFT can make outputs too narrow, hurting diversity.

## What I took away

- SFT should be judged by how well it sets up RL, not only by its own loss.
- Dropping already-learned examples from the gradient can preserve diversity.
- Track pass@k for k greater than 1 when choosing SFT checkpoints.

## Reference

- Original post: [https://x.com/giffmana/status/2098681993156255952](https://x.com/giffmana/status/2098681993156255952)
- Author: [@giffmana](https://x.com/giffmana) on X, posted 2026-09-12
