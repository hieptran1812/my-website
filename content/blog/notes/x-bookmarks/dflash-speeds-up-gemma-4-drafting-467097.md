---
title: "DFlash Speeds Up Gemma 4 Drafting"
publishDate: "2026-05-06"
date: "2026-05-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "speculative-decoding", "llm-inference", "open-models"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2051900751673467097-1.jpg"
excerpt: "The author says DFlash gives up to 6x faster Gemma 4 inference, building on Google's MTP drafters claiming 3x."
---

> Source: [@zhijianliu on X](https://x.com/zhijianliu/status/2051900751673467097) · posted 2026-05-06

![DFlash Speeds Up Gemma 4 Drafting (1)](/imgs/blogs/x-bookmarks/2051900751673467097-1.jpg)

![DFlash Speeds Up Gemma 4 Drafting (2)](/imgs/blogs/x-bookmarks/2051900751673467097-2.jpg)

## Summary

The author claims DFlash for Gemma 4 is up to 6x faster, open source with the same quality. They note MTP landing natively in Gemma 4. The quoted Google post says MTP drafters let Gemma 4 predict multiple tokens at once for up to 3x speed without losing intelligence.

## Background

MTP is multi-token prediction. Drafters guess several tokens ahead, and the main model verifies them, so output is the same but faster. The speed numbers are the authors' claims.

## What I took away

- Drafter-based methods speed up inference without changing outputs.
- Compare claimed speedups on your own hardware.

## Reference

- Original post: [https://x.com/zhijianliu/status/2051900751673467097](https://x.com/zhijianliu/status/2051900751673467097)
- Author: [@zhijianliu](https://x.com/zhijianliu) on X, posted 2026-05-06
