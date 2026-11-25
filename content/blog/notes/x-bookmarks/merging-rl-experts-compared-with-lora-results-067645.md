---
title: "Merging RL Experts Compared With LoRA Results"
publishDate: "2025-12-14"
date: "2025-12-14"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2000075364434067645-1.jpg"
excerpt: "The author describes an unexpected result from merging expert models before multi-task RL, in reply to a post on LoRA versus full fine-tuning."
---

> Source: [@FanqingMengAI on X](https://x.com/FanqingMengAI/status/2000075364434067645) · posted 2025-12-14

![Merging RL Experts Compared With LoRA Results (1)](/imgs/blogs/x-bookmarks/2000075364434067645-1.jpg)

![Merging RL Experts Compared With LoRA Results (2)](/imgs/blogs/x-bookmarks/2000075364434067645-2.png)

## Summary

@FanqingMengAI says they found something strange. They planned to train expert models for each sub-task, merge them, and continue with multi-task RL, but their ReasonGym experiments showed something unexpected. The quoted post is from Thinking Machines saying LoRA often matches full fine-tuning more closely than expected.

## Background

Full fine-tuning updates all weights, while LoRA updates a small add-on. The Thinking Machines post says results are often close. The author is exploring how that interacts with merging and RL.

## What I took away

- LoRA can match full fine-tuning in many cases.
- Test merging pipelines with both LoRA and full updates.
- Follow the thread for the later observations.

## Reference

- Original post: [https://x.com/FanqingMengAI/status/2000075364434067645](https://x.com/FanqingMengAI/status/2000075364434067645)
- Author: [@FanqingMengAI](https://x.com/FanqingMengAI) on X, posted 2025-12-14
