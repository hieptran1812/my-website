---
title: "StableReinforce for Stable Off-Policy RL on Tinker"
publishDate: "2026-10-21"
date: "2026-10-21"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training", "tools"]
author: "Hiep Tran"
featured: false
excerpt: "Tinker introduces StableReinforce, a loss method that aims to keep off-policy RL runs from collapsing."
---

> Source: [@tinkerapi on X](https://x.com/tinkerapi/status/2113019700661862805) · posted 2026-10-21

## Summary

@tinkerapi says off-policy RL is hard because runs can suddenly collapse due to staleness in async RL or numerical mismatch. They introduce StableReinforce, a method on Tinker meant to keep RL runs stable even when the learning policy drifts from the sampling policy.

## Background

In async RL, one set of GPUs generates samples while another trains, so samples come from a slightly older model. This is off-policy data. Also, the inference engine and trainer can compute slightly different token probabilities. Both effects can make training unstable unless the loss corrects for them.

## What I took away

- Mismatch between sampler and trainer is a common cause of RL collapse.
- Watch for sudden reward drops in async RL runs.
- Read the Tinker docs page to see how the correction works.

## Reference

- Original post: [https://x.com/tinkerapi/status/2113019700661862805](https://x.com/tinkerapi/status/2113019700661862805)
- Author: [@tinkerapi](https://x.com/tinkerapi) on X, posted 2026-10-21
