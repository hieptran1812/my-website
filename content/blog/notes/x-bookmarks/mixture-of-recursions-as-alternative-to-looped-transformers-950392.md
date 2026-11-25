---
title: "Mixture-of-Recursions As Alternative To Looped Transformers"
publishDate: "2026-04-17"
date: "2026-04-17"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "moe", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2045168844658950392-1.jpg"
excerpt: "The author argues Mythos should use Mixture-of-Recursions rather than uniform looped transformer layers."
---

> Source: [@reza_byt on X](https://x.com/reza_byt/status/2045168844658950392) · posted 2026-04-17

![Mixture-of-Recursions As Alternative To Looped Transformers](/imgs/blogs/x-bookmarks/2045168844658950392-1.jpg)

## Summary

The author reacts that Mythos is a looped transformer and says it should be Mixture-of-Recursions (MoR), which they claim is 2x faster with controlled effort. They compare it to the shift from dense to sparse MoE in 2023 and share a paper reading list.

## Background

A looped transformer reuses the same layers several times. MoR decides how many loops each token gets, so easy tokens use less compute. This is like MoE, where only some experts run.

## What I took away

- Adaptive depth is analogous to sparse MoE.
- The 2x claim is the author's.
- Follow the reading list.

## Reference

- Original post: [https://x.com/reza_byt/status/2045168844658950392](https://x.com/reza_byt/status/2045168844658950392)
- Author: [@reza_byt](https://x.com/reza_byt) on X, posted 2026-04-17
