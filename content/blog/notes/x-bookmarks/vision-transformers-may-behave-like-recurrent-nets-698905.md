---
title: "Vision Transformers May Behave Like Recurrent Nets"
publishDate: "2026-02-10"
date: "2026-02-10"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-research", "interpretability", "llm-architecture"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2021269389144698905-1.jpg"
excerpt: "A paper proposes the Block-Recurrent Hypothesis: a 2-block recurrent transformer recovers 96% of DINOv2 accuracy."
---

> Source: [@mozesjacobs on X](https://x.com/mozesjacobs/status/2021269389144698905) · posted 2026-02-10

![Vision Transformers May Behave Like Recurrent Nets](/imgs/blogs/x-bookmarks/2021269389144698905-1.jpg)

## Summary

The authors ask whether ViTs are secretly RNNs. Their 2-block recurrent transformer recovers 96% of DINOv2's ImageNet-1k accuracy and reproduces its activations one to one, which motivates the **Block-Recurrent Hypothesis**. The paper is at ICLR 2026.

## Background

A ViT stacks many different layers. A recurrent model reuses the same blocks repeatedly. If a few repeated blocks can mimic a deep stack, it suggests the layers do similar work, which could simplify models and help interpretation.

## What I took away

- Deep stacks may hide repeated computation.
- Weight sharing could shrink models with little loss.
- Read the arXiv paper for the exact claims.

## Reference

- Original post: [https://x.com/mozesjacobs/status/2021269389144698905](https://x.com/mozesjacobs/status/2021269389144698905)
- Author: [@mozesjacobs](https://x.com/mozesjacobs) on X, posted 2026-02-10
