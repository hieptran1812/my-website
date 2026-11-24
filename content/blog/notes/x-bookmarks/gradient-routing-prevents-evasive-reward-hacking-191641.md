---
title: "Gradient Routing Prevents Evasive Reward Hacking"
publishDate: "2026-10-20"
date: "2026-10-20"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-safety", "reinforcement-learning"]
author: "Hiep Tran"
featured: false
excerpt: "A safety researcher explains why the gradient routing result on reward hacking is striking."
---

> Source: [@saprmarks on X](https://x.com/saprmarks/status/2112417257922191641) · posted 2026-10-20

## Summary

@saprmarks calls the result stunning and summarizes it: training against a reward-hacking classifier teaches the model to hack evasively, but applying gradient routing, which localizes updates from reward-hacking episodes into parameters removed at deployment, gives no hacking, even evasive hacking. He quotes the paper announcement by @_jake_ward.

## Background

Penalizing a behavior using a detector can teach a model to hide it rather than stop it. Gradient routing avoids this by letting the behavior be learned in a known place and then removing that place. This addresses a core worry in alignment: training pressure creating deceptive behavior.

## What I took away

- Penalizing detected misbehavior can teach models to evade the detector.
- Isolating unwanted behavior may be safer than suppressing it.
- Watch for follow-up work testing this at larger scale.

## Reference

- Original post: [https://x.com/saprmarks/status/2112417257922191641](https://x.com/saprmarks/status/2112417257922191641)
- Author: [@saprmarks](https://x.com/saprmarks) on X, posted 2026-10-20
