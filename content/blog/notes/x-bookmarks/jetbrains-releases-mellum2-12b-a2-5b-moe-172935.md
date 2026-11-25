---
title: "JetBrains Releases Mellum2 12B A2.5B MoE"
publishDate: "2026-06-01"
date: "2026-06-01"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "open-models", "moe", "pretraining"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2061438808290172935-1.jpg"
excerpt: "The team releases Mellum2, a 12B MoE with 2.5B active parameters, pre-trained on about 11T tokens."
---

> Source: [@nv_pavlichenko on X](https://x.com/nv_pavlichenko/status/2061438808290172935) · posted 2026-06-01

![JetBrains Releases Mellum2 12B A2.5B MoE](/imgs/blogs/x-bookmarks/2061438808290172935-1.jpg)

## Summary

The author announces Mellum2, their first serious LLM. It is a 12B A2.5B MoE LLM pre-trained on about 11T tokens and post-trained with RLVR. They led the team for six months and release base, SFT and RL checkpoints along with a tech report.

## Background

MoE (mixture of experts) activates only some parameters per token, so a 12B model costs about as much as 2.5B to run. RLVR is RL with verifiable rewards.

## What I took away

- Open checkpoints at each training stage.
- Small active parameter counts keep inference cheap.

## Reference

- Original post: [https://x.com/nv_pavlichenko/status/2061438808290172935](https://x.com/nv_pavlichenko/status/2061438808290172935)
- Author: [@nv_pavlichenko](https://x.com/nv_pavlichenko) on X, posted 2026-06-01
