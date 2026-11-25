---
title: "True On-Policy RL for VLMs in Miles"
publishDate: "2025-12-12"
date: "2025-12-12"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "serving", "llm-inference"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1999548119034564673-1.png"
excerpt: "Miles extends true on-policy training to vision language models with FSDP and SGLang."
---

> Source: [@ying11231 on X](https://x.com/ying11231/status/1999548119034564673) · posted 2025-12-12

![True On-Policy RL for VLMs in Miles](/imgs/blogs/x-bookmarks/1999548119034564673-1.png)

## Summary

@ying11231 says Miles now extends true on-policy training to VLMs for the FSDP backend and SGLang. The quoted post says the team earlier matched precision between FSDP and SGLang so SGLang log probs equal FSDP log probs.

## Background

On-policy means the data comes from the same model being trained. When the inference engine and trainer compute slightly different log probs, training drifts. Matching them exactly keeps RL stable.

## What I took away

- Mismatch between inference and training engines hurts RL.
- Exact log prob alignment is a goal for on-policy training.
- Support now covers vision models.

## Reference

- Original post: [https://x.com/ying11231/status/1999548119034564673](https://x.com/ying11231/status/1999548119034564673)
- Author: [@ying11231](https://x.com/ying11231) on X, posted 2025-12-12
