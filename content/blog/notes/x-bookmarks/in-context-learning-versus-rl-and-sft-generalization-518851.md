---
title: "In-Context Learning Versus RL And SFT Generalization"
publishDate: "2026-06-26"
date: "2026-06-26"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "post-training", "reinforcement-learning", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2070599709962518851-1.jpg"
excerpt: "A reaction to a paper comparing how RL, in-context learning and SFT generalize across task distances."
---

> Source: [@arb8020 on X](https://x.com/arb8020/status/2070599709962518851) · posted 2026-06-26

![In-Context Learning Versus RL And SFT Generalization (1)](/imgs/blogs/x-bookmarks/2070599709962518851-1.jpg)

![In-Context Learning Versus RL And SFT Generalization (2)](/imgs/blogs/x-bookmarks/2070599709962518851-2.jpg)

## Summary

@arb8020 says ICL may usually be all you need, with RL adding greater generalization, and jokes that SFT is in shambles. The quoted paper (arXiv 2606.25450) studies how RL, ICL and SFT generalize across task distances. The quote adds that RL has the best transfer characteristics.

## Background

ICL (in-context learning) means giving examples in the prompt without changing weights. SFT is supervised fine-tuning on examples. RL trains from reward signals. The paper compares how far each carries to tasks unlike the training ones. The author says they still need to dig into it.

## What I took away

- Try prompting with examples before fine-tuning.
- RL is claimed to transfer best to distant tasks.
- Read the paper before relying on the tweet-level claim.

## Reference

- Original post: [https://x.com/arb8020/status/2070599709962518851](https://x.com/arb8020/status/2070599709962518851)
- Author: [@arb8020](https://x.com/arb8020) on X, posted 2026-06-26
