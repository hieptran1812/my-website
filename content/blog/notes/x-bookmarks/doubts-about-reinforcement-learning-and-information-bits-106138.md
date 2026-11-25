---
title: "Doubts About Reinforcement Learning And Information Bits"
publishDate: "2025-11-27"
date: "2025-11-27"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "pretraining", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1994159865196106138-1.jpg"
excerpt: "A reply agrees it is hard to believe current RL leads to AGI, quoting an argument about information per rollout."
---

> Source: [@clu_cheng on X](https://x.com/clu_cheng/status/1994159865196106138) · posted 2025-11-27

![Doubts About Reinforcement Learning And Information Bits](/imgs/blogs/x-bookmarks/1994159865196106138-1.jpg)

## Summary

The author says they feel the same and find it hard to believe that current RL is the right way to AGI. The quoted post, based on ideas from Dwarkesh's post, says early pretraining gives about 18 bits of information per forward pass for a 256k vocabulary, while an RL rollout of 32k tokens is compared against that. The quoted text is cut off.

## Background

In pretraining, each predicted token gives a rich learning signal. In RL, a long rollout often yields only one reward at the end, so far less information per token generated. That argument suggests RL is sample-inefficient compared with pretraining.

## What I took away

- RL signals are sparse compared with next-token prediction.
- Information per sample is a useful way to compare training methods.
- The argument is partial here, so read the full thread.

## Reference

- Original post: [https://x.com/clu_cheng/status/1994159865196106138](https://x.com/clu_cheng/status/1994159865196106138)
- Author: [@clu_cheng](https://x.com/clu_cheng) on X, posted 2025-11-27
