---
title: "Eureka Machine Sets Backprop-Free Training Record"
publishDate: "2026-11-24"
date: "2026-11-24"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-research", "pretraining", "tools"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2125263417040846932-1.jpg"
excerpt: "A post says the Eureka Machine set a new record on a backprop-free LLM training speedrun, reaching 3.5 val loss in 9.85 seconds."
---

> Source: [@tydsh on X](https://x.com/tydsh/status/2125263417040846932) · posted 2026-11-24

![Eureka Machine Sets Backprop-Free Training Record](/imgs/blogs/x-bookmarks/2125263417040846932-1.jpg)

## Summary

The author says the Eureka Machine sets a new record for backprop-free LLM training and credits the team. The quoted post describes the noise speedrun benchmark, where the goal is to reach 3.5 validation loss as fast as possible, and says the record dropped from over 14 seconds to 9.85 seconds.

## Background

Backpropagation is the usual way to compute gradients for training. Backprop-free methods avoid it and are interesting because they can be cheaper or simpler. Speedruns give a fixed target so progress is easy to compare.

## What I took away

- Speedrun benchmarks make research progress measurable.
- Backprop-free training is still an active research area.
- See the GitHub repo for the discoveries.

## Reference

- Original post: [https://x.com/tydsh/status/2125263417040846932](https://x.com/tydsh/status/2125263417040846932)
- Author: [@tydsh](https://x.com/tydsh) on X, posted 2026-11-24
