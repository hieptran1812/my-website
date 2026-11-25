---
title: "Mixup-Style Perturbation Improves Pretraining Distillation"
publishDate: "2026-11-16"
date: "2026-11-16"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "distillation", "pretraining", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2122428389684588624-1.jpg"
excerpt: "A researcher notes a new paper applies a mixup-like input perturbation to language model distillation, echoing earlier vision work."
---

> Source: [@giffmana on X](https://x.com/giffmana/status/2122428389684588624) · posted 2026-11-16

![Mixup-Style Perturbation Improves Pretraining Distillation (1)](/imgs/blogs/x-bookmarks/2122428389684588624-1.jpg)

![Mixup-Style Perturbation Improves Pretraining Distillation (2)](/imgs/blogs/x-bookmarks/2122428389684588624-2.png)

## Summary

The author says the paper did their distillation thing for language models, with the equivalent of mixup. They note it applies the perturbation to only part of distillation and finishes on pure examples, which they think they should have done earlier. The quoted paper says it perturbs inputs for the first part of training in pretraining distillation.

## Background

Distillation trains a small student to match a bigger teacher's predictions. Mixup blends inputs to regularize training. Ending on clean data helps the student fit the real distribution.

## What I took away

- Perturb inputs early, finish on clean data.
- Vision ideas often transfer to language.
- Check the paper for gains.

## Reference

- Original post: [https://x.com/giffmana/status/2122428389684588624](https://x.com/giffmana/status/2122428389684588624)
- Author: [@giffmana](https://x.com/giffmana) on X, posted 2026-11-16
