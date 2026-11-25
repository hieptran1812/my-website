---
title: "Open Models Doing Research-Level Math Within a Year"
publishDate: "2026-11-24"
date: "2026-11-24"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "math", "open-models"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2125243119935332618-1.jpg"
excerpt: "A post predicts open models will do research-level mathematics within a year, quoting RL results on a 2B proof model."
---

> Source: [@_lewtun on X](https://x.com/_lewtun/status/2125243119935332618) · posted 2026-11-24

![Open Models Doing Research-Level Math Within a Year](/imgs/blogs/x-bookmarks/2125243119935332618-1.jpg)

## Summary

The author thinks that within one year open models will also be able to do research-level mathematics. The quoted post says a 2B model was trained with RL to write better math proofs using a custom harness, improving from 0.22 to 0.41 on held-out proofs with no SFT, using a roughly 280-line harness and Hugging Face infrastructure.

## Background

A harness is the code that sets up tasks and scoring for a model. Training with RL from a base model means the model learns from reward only, without example solutions. A small model improving on proofs suggests the approach scales, though this is an early result.

## What I took away

- Small models can improve on proofs with RL alone.
- A small harness can be enough to run such training.
- The prediction is an opinion, not a result.

## Reference

- Original post: [https://x.com/_lewtun/status/2125243119935332618](https://x.com/_lewtun/status/2125243119935332618)
- Author: [@_lewtun](https://x.com/_lewtun) on X, posted 2026-11-24
