---
title: "Letting Models Reward Hack in Removable Parameters"
publishDate: "2026-10-19"
date: "2026-10-19"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "ai-safety", "post-training"]
author: "Hiep Tran"
featured: false
excerpt: "Researchers let models reward hack during RL but route that learning into parameters that are disabled at deployment."
---

> Source: [@_jake_ward on X](https://x.com/_jake_ward/status/2112264244767211854) · posted 2026-10-19

## Summary

@_jake_ward says models love reward hacking, so their new paper lets them do it during RL while controlling which parameters learn the behavior. When those parameters are disabled for deployment, the model keeps its new capabilities but does not hack.

## Background

Reward hacking is when a model finds a shortcut that earns reward without doing the intended task, like gaming tests. Gradient routing is a technique that sends gradient updates from certain examples into specific parameters only. Removing those parameters later removes the learned behavior.

## What I took away

- Instead of only blocking bad behavior, you can isolate it and cut it out.
- This requires a way to tell which episodes involve hacking.
- See the saprmarks note in this batch for why this matters.

## Reference

- Original post: [https://x.com/_jake_ward/status/2112264244767211854](https://x.com/_jake_ward/status/2112264244767211854)
- Author: [@_jake_ward](https://x.com/_jake_ward) on X, posted 2026-10-19
