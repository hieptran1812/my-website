---
title: "Linear Attention And State Space Models Converge"
publishDate: "2026-07-08"
date: "2026-07-08"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "learning-resources"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2074683873859142094-1.jpg"
excerpt: "A note that linear attention and state-space models are convergent evolution, with a recommended video."
---

> Source: [@akashmjn on X](https://x.com/akashmjn/status/2074683873859142094) · posted 2026-07-08

![Linear Attention And State Space Models Converge (1)](/imgs/blogs/x-bookmarks/2074683873859142094-1.jpg)

![Linear Attention And State Space Models Converge (2)](/imgs/blogs/x-bookmarks/2074683873859142094-2.jpg)

## Summary

The author says linear attention and state-space models are convergent evolution, citing Qwen 3.5 with Gated Delta Net hybrid and Nemotron 3 with a Mamba hybrid. The quoted post is a video explaining linear attention and its link to older and modern approaches.

## Background

Both families keep a fixed-size state updated per token instead of a growing KV cache, so they are cheaper for long context. Hybrids mix them with normal attention.

## What I took away

- Linear attention and SSMs are related.
- Hybrids are common in new models.
- Watch the video for intuition.

## Reference

- Original post: [https://x.com/akashmjn/status/2074683873859142094](https://x.com/akashmjn/status/2074683873859142094)
- Author: [@akashmjn](https://x.com/akashmjn) on X, posted 2026-07-08
