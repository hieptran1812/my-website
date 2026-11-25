---
title: "Why Marin 535B-A23B Is Designed That Way"
publishDate: "2026-11-10"
date: "2026-11-10"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "moe", "pretraining"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2120034819291898364-1.jpg"
excerpt: "Percy Liang points to a writeup explaining design decisions behind the Marin 535B-A23B model."
---

> Source: [@percyliang on X](https://x.com/percyliang/status/2120034819291898364) · posted 2026-11-10

![Why Marin 535B-A23B Is Designed That Way (1)](/imgs/blogs/x-bookmarks/2120034819291898364-1.jpg)

![Why Marin 535B-A23B Is Designed That Way (2)](/imgs/blogs/x-bookmarks/2120034819291898364-2.jpg)

## Summary

The author asks why Marin 535B-A23B has its size, 8 of 384 experts, GatedNorm, sconv, XSA and a linear learning rate. The answer is tradeoffs among timelines, compute efficiency and MoE router stability. The quoted post says the model finished pretraining on 18T tokens and links the architecture rationale.

## Background

In a mixture-of-experts model, only some experts run per token, so 535B total parameters with 23B active is cheaper to run. Router stability means keeping expert selection from collapsing during training. Documented rationale is valuable.

## What I took away

- Real architecture choices reflect schedule and stability constraints.
- Read the writeup for decisions, not only results.
- Open design notes help others reproduce.

## Reference

- Original post: [https://x.com/percyliang/status/2120034819291898364](https://x.com/percyliang/status/2120034819291898364)
- Author: [@percyliang](https://x.com/percyliang) on X, posted 2026-11-10
