---
title: "Autoresearch At Home Shared Agent Network"
publishDate: "2026-03-11"
date: "2026-03-11"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-agents", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2031873645845643367-1.jpg"
excerpt: "A project inspired by Karpathy's autoresearch lets any agent on the internet join collaborative AI/ML research."
---

> Source: [@christinetyip on X](https://x.com/christinetyip/status/2031873645845643367) · posted 2026-03-11

![Autoresearch At Home Shared Agent Network (1)](/imgs/blogs/x-bookmarks/2031873645845643367-1.jpg)

![Autoresearch At Home Shared Agent Network (2)](/imgs/blogs/x-bookmarks/2031873645845643367-2.jpg)

![Autoresearch At Home Shared Agent Network (3)](/imgs/blogs/x-bookmarks/2031873645845643367-3.jpg)

## Summary

The author says they built **autoresearch@home**, where any agent on the internet can join and collaborate on AI/ML research, so hundreds or thousands can explore the search space together. The quoted Karpathy post says autoresearch ran about 2 days on a depth=12 nanochat model and found about 20 changes that improved validation loss.

## Background

Autoresearch lets an agent try training tweaks automatically and keep those that lower validation loss. Spreading this across many agents is like parallel hyperparameter search shared by a community.

## What I took away

- Parallel agents can cover a research search space faster than one.
- Karpathy reports the found gains were additive and transferred to larger models.
- Shared experiment logs matter so agents do not repeat work.

## Reference

- Original post: [https://x.com/christinetyip/status/2031873645845643367](https://x.com/christinetyip/status/2031873645845643367)
- Author: [@christinetyip](https://x.com/christinetyip) on X, posted 2026-03-11
