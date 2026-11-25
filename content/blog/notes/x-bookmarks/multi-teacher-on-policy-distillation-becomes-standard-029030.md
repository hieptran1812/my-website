---
title: "Multi-Teacher On-Policy Distillation Becomes Standard"
publishDate: "2026-06-04"
date: "2026-06-04"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "distillation", "post-training", "reinforcement-learning"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2062528878997029030-1.jpg"
excerpt: "Nathan Lambert says Nvidia joined the multi-teacher on-policy distillation approach, now an industry norm."
---

> Source: [@natolambert on X](https://x.com/natolambert/status/2062528878997029030) · posted 2026-06-04

![Multi-Teacher On-Policy Distillation Becomes Standard (1)](/imgs/blogs/x-bookmarks/2062528878997029030-1.jpg)

![Multi-Teacher On-Policy Distillation Becomes Standard (2)](/imgs/blogs/x-bookmarks/2062528878997029030-2.jpg)

## Summary

The author says Nvidia joined the multi-teacher on-policy distillation (MODP) gang, which he calls industry standard post-training now. He says multi-teacher SFT to RL, done by Microsoft in their first model, was set by DeepSeek R1, and expects MAI 2 to use MODP. The quoted post describes Nvidia's pipeline: SFT, multi-environment RLVR, then multi-teacher on-policy distillation.

## Background

RLVR is reinforcement learning with verifiable rewards, such as checking code or math answers. In multi-teacher on-policy distillation, one student learns from several specialist teachers on its own generated outputs.

## What I took away

- Recipes now mix SFT, RLVR and distillation.
- The claim about industry standard is the author's view.

## Reference

- Original post: [https://x.com/natolambert/status/2062528878997029030](https://x.com/natolambert/status/2062528878997029030)
- Author: [@natolambert](https://x.com/natolambert) on X, posted 2026-06-04
