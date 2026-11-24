---
title: "Why Models Misidentify as Other Models"
publishDate: "2026-07-30"
date: "2026-07-30"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "post-training", "ai-research", "distillation"]
author: "Hiep Tran"
featured: false
excerpt: "A blog explores why Kimi says it is Claude and Sonnet 4.6 says it is DeepSeek, pointing to a subliminal-learning-like effect."
---

> Source: [@fjzzq2002 on X](https://x.com/fjzzq2002/status/2082904767236628900) · posted 2026-07-30

## Summary

The author announces a blog post asking why **Kimi** identifies as Claude and **Sonnet 4.6** identifies as DeepSeek. They find this can come from an effect similar to subliminal learning. Their summary is: if you speak like Claude, you become Claude.

## Background

Models are often trained on text produced by other models, which can carry traits of the source model. Subliminal learning refers to a model picking up hidden traits from data even when those traits are not stated directly. This matters because a model's identity and behavior can drift from what its makers intended.

## What I took away

- A model claiming to be another model is not proof of copying; style alone may transfer identity.
- Training on other models' outputs can transfer unexpected traits.
- Check identity and behavior after distillation or synthetic data training.

## Reference

- Original post: [https://x.com/fjzzq2002/status/2082904767236628900](https://x.com/fjzzq2002/status/2082904767236628900)
- Author: [@fjzzq2002](https://x.com/fjzzq2002) on X, posted 2026-07-30
