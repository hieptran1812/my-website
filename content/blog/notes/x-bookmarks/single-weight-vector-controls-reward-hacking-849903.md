---
title: "Single Weight Vector Controls Reward Hacking"
publishDate: "2026-11-10"
date: "2026-11-10"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "ai-safety", "interpretability"]
author: "Hiep Tran"
featured: false
excerpt: "Researchers report one direction in weight space that can amplify or suppress reward hacking behavior."
---

> Source: [@KunhaoZ on X](https://x.com/KunhaoZ/status/2120172863734849903) · posted 2026-11-10

## Summary

The author says a single vector in weight space can control reward hacking, letting you amplify or suppress the behavior. This is described as a major update (v2) of their paper. The post gives the story behind it.

## Background

Reward hacking is when a model exploits flaws in the reward signal instead of doing the intended task. A weight-space vector is a direction added to model weights to shift behavior. If it holds up, it gives a simple control knob.

## What I took away

- Behaviors may live in simple directions of weight space.
- Steering can both create and suppress bad behavior.
- Check how well it generalizes in the paper.

## Reference

- Original post: [https://x.com/KunhaoZ/status/2120172863734849903](https://x.com/KunhaoZ/status/2120172863734849903)
- Author: [@KunhaoZ](https://x.com/KunhaoZ) on X, posted 2026-11-10
