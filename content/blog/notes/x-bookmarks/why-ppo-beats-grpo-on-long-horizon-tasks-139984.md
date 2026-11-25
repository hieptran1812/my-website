---
title: "Why PPO Beats GRPO On Long-Horizon Tasks"
publishDate: "2026-06-21"
date: "2026-06-21"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2068544629709139984-1.jpg"
excerpt: "A post argues PPO suits long-horizon RL better because group synchronization strains infrastructure."
---

> Source: [@QPHutu on X](https://x.com/QPHutu/status/2068544629709139984) · posted 2026-06-21

![Why PPO Beats GRPO On Long-Horizon Tasks](/imgs/blogs/x-bookmarks/2068544629709139984-1.jpg)

## Summary

@QPHutu gives reasons PPO is better than GRPO for long-horizon tasks. First, group-level synchronization is hard for infra in both sync (long tail) and async (staleness) cases. Second, group or critic is just variance reduction. The text is cut off.

## Background

GRPO needs several samples per prompt to finish before updating, so slow ones delay everyone. With long tasks the lengths vary a lot. A critic avoids that waiting.

## What I took away

- Infra constraints can decide the RL algorithm.
- Group baselines are one of several variance reduction options.

## Reference

- Original post: [https://x.com/QPHutu/status/2068544629709139984](https://x.com/QPHutu/status/2068544629709139984)
- Author: [@QPHutu](https://x.com/QPHutu) on X, posted 2026-06-21
