---
title: "Diagnosing Training Setups From Loss Curves"
publishDate: "2025-12-14"
date: "2025-12-14"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "ai-research", "learning-resources"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2000082772338311441-1.jpg"
excerpt: "A joke about a trivia night where you diagnose training problems from loss curves, quoting a post that says batch size is way too small."
---

> Source: [@sashadoubov on X](https://x.com/sashadoubov/status/2000082772338311441) · posted 2025-12-14

![Diagnosing Training Setups From Loss Curves](/imgs/blogs/x-bookmarks/2000082772338311441-1.jpg)

## Summary

@sashadoubov jokes about a trivia night built around diagnosing training setups from loss curves. The quoted post simply says that your batch size is way too small.

## Background

A loss curve shows model error over training steps. Noisy or unstable curves often point to settings like a small batch size or a high learning rate. Experienced practitioners can guess causes from curve shapes.

## What I took away

- Noisy loss curves can indicate too small a batch size.
- Practice reading loss curves to debug faster.
- Change one setting at a time to confirm a diagnosis.

## Reference

- Original post: [https://x.com/sashadoubov/status/2000082772338311441](https://x.com/sashadoubov/status/2000082772338311441)
- Author: [@sashadoubov](https://x.com/sashadoubov) on X, posted 2025-12-14
