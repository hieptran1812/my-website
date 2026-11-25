---
title: "Training Qwen3-Coder To Use A Debugger"
publishDate: "2026-04-28"
date: "2026-04-28"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "post-training", "ai-agents", "reinforcement-learning"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2049192929739280482-1.jpg"
excerpt: "The author post-trained Qwen3-Coder to fix bugs with a real debugger, claiming solve rate rose from 70% to 89%."
---

> Source: [@moofeez on X](https://x.com/moofeez/status/2049192929739280482) · posted 2026-04-28

![Training Qwen3-Coder To Use A Debugger](/imgs/blogs/x-bookmarks/2049192929739280482-1.jpg)

## Summary

The author says they post-trained Qwen3-Coder to fix bugs using an actual debugger. They report the solve rate going from 70% to 89% and median turns to fix dropping from 46 to 19. The model reasons from execution, inspects live variables and call stacks and sets breakpoints.

## Background

Most coding models read code or add print statements. A debugger gives real runtime state. Learning to use it can reduce guessing. Numbers are the author's own.

## What I took away

- Giving agents runtime tools can cut wasted turns.
- Check results on bugs outside the training set.

## Reference

- Original post: [https://x.com/moofeez/status/2049192929739280482](https://x.com/moofeez/status/2049192929739280482)
- Author: [@moofeez](https://x.com/moofeez) on X, posted 2026-04-28
