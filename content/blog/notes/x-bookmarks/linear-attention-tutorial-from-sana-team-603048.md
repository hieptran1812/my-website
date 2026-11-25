---
title: "Linear Attention Tutorial From SANA Team"
publishDate: "2025-12-25"
date: "2025-12-25"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "diffusion", "learning-resources"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2004223136049603048-1.jpg"
excerpt: "A tutorial from the SANA team explains linear attention, with constant memory and long context."
---

> Source: [@MikeShou1 on X](https://x.com/MikeShou1/status/2004223136049603048) · posted 2025-12-25

![Linear Attention Tutorial From SANA Team](/imgs/blogs/x-bookmarks/2004223136049603048-1.jpg)

## Summary

Mike Shou praises a comprehensive tutorial on linear attention and its variants. The quoted SANA team post says linear attention gives global awareness with constant memory by accumulating state. It also covers the evolution from softmax attention.

## Background

Standard softmax attention costs more as sequences grow. Linear attention keeps a running state so memory stays fixed while the model reads more. This allows very long contexts but can lose some precision.

## What I took away

- Linear attention trades exact lookup for constant memory.
- Read the tutorial to see how variants differ.

## Reference

- Original post: [https://x.com/MikeShou1/status/2004223136049603048](https://x.com/MikeShou1/status/2004223136049603048)
- Author: [@MikeShou1](https://x.com/MikeShou1) on X, posted 2025-12-25
