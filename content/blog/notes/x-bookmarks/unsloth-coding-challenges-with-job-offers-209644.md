---
title: "Unsloth Coding Challenges With Job Offers"
publishDate: "2025-02-16"
date: "2025-02-16"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "career", "llm-inference", "post-training"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1891194528931209644-1.jpg"
excerpt: "Unsloth announces five engineering challenges, where a high enough score earns a job offer, and quotes a fix for a gradient accumulation bug."
---

> Source: [@danielhanchen on X](https://x.com/danielhanchen/status/1891194528931209644) · posted 2025-02-16

![Unsloth Coding Challenges With Job Offers (1)](/imgs/blogs/x-bookmarks/1891194528931209644-1.jpg)

![Unsloth Coding Challenges With Job Offers (2)](/imgs/blogs/x-bookmarks/1891194528931209644-2.jpg)

## Summary

Daniel Han says Unsloth made 5 challenges. Scoring 47 points earns an offer of $400K to $500K a year plus equity as Founding Engineer, and 32 points leads to an ML Engineer role at $250K to $300K. He says no experience or PhD is needed. The first challenge is converting nf4 / BnB 4bit weights. The quoted post describes a fix for a bug that made training losses diverge with large gradient accumulation sizes.

## Background

Gradient accumulation (GA) adds up gradients over several small batches before one update, so it should match training on one big batch. If the loss is normalized wrongly, the results differ. nf4 and BnB 4bit are low-bit weight formats used to fit big models in less memory.

## What I took away

- Open challenges can be a way into a job without a formal background.
- Gradient accumulation should match full batch training mathematically; if losses differ, suspect a normalization bug.
- Check loss curves against a full-batch baseline when you change batch setup.

## Reference

- Original post: [https://x.com/danielhanchen/status/1891194528931209644](https://x.com/danielhanchen/status/1891194528931209644)
- Author: [@danielhanchen](https://x.com/danielhanchen) on X, posted 2025-02-16
