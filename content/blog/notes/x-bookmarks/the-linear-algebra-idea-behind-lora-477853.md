---
title: "The Linear Algebra Idea Behind LoRA"
publishDate: "2025-11-16"
date: "2025-11-16"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "post-training", "math", "llm-architecture"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1989846655974477853-1.png"
excerpt: "The author notes that LoRA rests on one simple linear algebra result yet is a highly effective way to finetune LLMs."
---

> Source: [@Creative_Math_ on X](https://x.com/Creative_Math_/status/1989846655974477853) · posted 2025-11-16

![The Linear Algebra Idea Behind LoRA](/imgs/blogs/x-bookmarks/1989846655974477853-1.png)

## Summary

The author finds it striking that the key idea behind LoRA is just a simple result in linear algebra. They say it gives one of the most effective ways to finetune LLMs. The image or thread with the exact result is at the link.

## Background

LoRA finetunes a model by adding a small low-rank update to a weight matrix instead of changing the whole matrix. A low-rank matrix can be written as the product of two thin matrices, which has far fewer parameters. This makes finetuning cheaper in memory and compute.

## What I took away

- Low-rank updates cut the number of trainable parameters a lot.
- Simple math can underlie a very practical technique.
- Refresh matrix rank and factorization if LoRA feels opaque.

## Reference

- Original post: [https://x.com/Creative_Math_/status/1989846655974477853](https://x.com/Creative_Math_/status/1989846655974477853)
- Author: [@Creative_Math_](https://x.com/Creative_Math_) on X, posted 2025-11-16
