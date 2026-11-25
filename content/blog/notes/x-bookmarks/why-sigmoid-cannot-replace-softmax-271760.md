---
title: "Why Sigmoid Cannot Replace Softmax"
publishDate: "2026-02-17"
date: "2026-02-17"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "learning-resources", "math", "llm-architecture"]
author: "Hiep Tran"
featured: false
excerpt: "prajdabre asks why replacing softmax with sigmoid fails an interview."
---

> Source: [@prajdabre on X](https://x.com/prajdabre/status/2023878727571271760) · posted 2026-02-17

## Summary

The question says that saying yes fails, even though both give values between 0 and 1, and asks why.

## Background

Softmax outputs sum to 1 and compete as a distribution; sigmoid treats each independently, so outputs need not sum to 1.

## What I took away

- Softmax gives a distribution.
- Sigmoid suits independent labels.

## Reference

- Original post: [https://x.com/prajdabre/status/2023878727571271760](https://x.com/prajdabre/status/2023878727571271760)
- Author: [@prajdabre](https://x.com/prajdabre) on X, posted 2026-02-17
