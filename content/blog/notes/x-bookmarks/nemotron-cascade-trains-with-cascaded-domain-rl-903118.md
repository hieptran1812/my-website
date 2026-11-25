---
title: "Nemotron-Cascade Trains With Cascaded Domain RL"
publishDate: "2025-12-16"
date: "2025-12-16"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training", "open-models"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2001024175058903118-1.jpg"
excerpt: "ychenNLP shares Nemotron-Cascade results from cascade RL across domains, including an IOI 2025 silver medal."
---

> Source: [@ychenNLP on X](https://x.com/ychenNLP/status/2001024175058903118) · posted 2025-12-16

![Nemotron-Cascade Trains With Cascaded Domain RL (1)](/imgs/blogs/x-bookmarks/2001024175058903118-1.jpg)

![Nemotron-Cascade Trains With Cascaded Domain RL (2)](/imgs/blogs/x-bookmarks/2001024175058903118-2.jpg)

## Summary

The author says the Nemotron-Cascade models earned a silver medal at IOI 2025 and outperform DeepSeek-R1-0528 on LiveCodeBench. Instead of mixing tasks, they scale Cascade RL in a curriculum from RLHF to Instruct to Math to Code to SWE. The quoted post introduces the Nemotron-Cascade family of general-purpose reasoning models.

## Background

Cascade RL trains one domain after another instead of all at once. Each stage can use its own reward and settings, which avoids conflicts between tasks such as chat style and code. IOI is the International Olympiad in Informatics, a coding contest. The performance claims come from the authors.

## What I took away

- Training domains in sequence can be simpler than mixing them.
- Order of stages matters, so test different curricula.
- Read the report for what they learned at each stage.

## Reference

- Original post: [https://x.com/ychenNLP/status/2001024175058903118](https://x.com/ychenNLP/status/2001024175058903118)
- Author: [@ychenNLP](https://x.com/ychenNLP) on X, posted 2025-12-16
