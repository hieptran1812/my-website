---
title: "Using Model Internals During Training"
publishDate: "2026-07-15"
date: "2026-07-15"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "interpretability", "ai-safety", "post-training"]
author: "Hiep Tran"
featured: false
excerpt: "A reaction to research that uses a model's internal representations in training, following a LessWrong post arguing it is reasonable."
---

> Source: [@teortaxesTex on X](https://x.com/teortaxesTex/status/2077414984473075867) · posted 2026-07-15

## Summary

The quoted post says the authors finally did research described in a LessWrong post titled It Is Reasonable To Research How To Use Model Internals In Training. The author reacts with the phrase Deepest Lore, an insider joke.

## Background

Interpretability tools read a model's internal activations to understand what it represents. Using these signals in training, for example to penalize certain internal features, is debated in AI safety because the model might learn to hide those features instead of changing behavior.

## What I took away

- Training on interpretability signals is powerful but may teach models to evade them.
- Read the LessWrong post for the arguments for and against.
- Open the link to see the actual research results.

## Reference

- Original post: [https://x.com/teortaxesTex/status/2077414984473075867](https://x.com/teortaxesTex/status/2077414984473075867)
- Author: [@teortaxesTex](https://x.com/teortaxesTex) on X, posted 2026-07-15
