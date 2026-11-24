---
title: "Inoculation Adapters for Selective Generalization"
publishDate: "2026-07-17"
date: "2026-07-17"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-safety", "post-training", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "A new paper proposes inoculation adapters to teach models desired traits while blocking undesired ones."
---

> Source: [@DanielCHTan97 on X](https://x.com/DanielCHTan97/status/2078129689932308961) · posted 2026-07-17

## Summary

The authors propose inoculation adapters (IA) for selective generalization. They report that across 9 empirical settings, IA beats baselines at teaching a model desirable traits while blocking undesirable ones.

## Background

When you fine-tune a model on data, it can pick up unwanted side behaviors along with the intended skill. Inoculation refers to deliberately exposing the model to the unwanted trait in a controlled way so it does not generalize it. An adapter is a small add-on module trained alongside or instead of the full model.

## What I took away

- Fine-tuning data can teach unintended traits; selective generalization tries to separate them.
- Adapters give a modular way to isolate behaviors.
- Read the thread for the 9 settings and what counts as a baseline.

## Reference

- Original post: [https://x.com/DanielCHTan97/status/2078129689932308961](https://x.com/DanielCHTan97/status/2078129689932308961)
- Author: [@DanielCHTan97](https://x.com/DanielCHTan97) on X, posted 2026-07-17
