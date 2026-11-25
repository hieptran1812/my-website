---
title: "Why Data Filtering Fails To Remove Traits"
publishDate: "2026-07-07"
date: "2026-07-07"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "post-training", "interpretability", "ai-safety"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2074572588182102193-1.jpg"
excerpt: "Neel Nanda notes that filtering training data to remove unwanted LLM behaviors worked worse than expected."
---

> Source: [@NeelNanda5 on X](https://x.com/NeelNanda5/status/2074572588182102193) · posted 2026-07-07

![Why Data Filtering Fails To Remove Traits](/imgs/blogs/x-bookmarks/2074572588182102193-1.jpg)

## Summary

@NeelNanda5 says he expected a benchmark of data attribution methods, but found that in realistic settings problems are hard to trace to specific data. The quoted post says that after SFT, filtering out suspected training data worked much worse than expected, and Olmo kept inheriting the traits.

## Background

Data attribution tries to find which training examples cause a model behavior. SFT means supervised fine-tuning on example outputs. The finding suggests behaviors are spread across data rather than sitting in a few bad examples.

## What I took away

- Do not assume an unwanted behavior can be fixed by deleting a few training examples.
- Traits may be diffusely encoded in the data.
- Attribution methods need testing in realistic, not toy, settings.

## Reference

- Original post: [https://x.com/NeelNanda5/status/2074572588182102193](https://x.com/NeelNanda5/status/2074572588182102193)
- Author: [@NeelNanda5](https://x.com/NeelNanda5) on X, posted 2026-07-07
