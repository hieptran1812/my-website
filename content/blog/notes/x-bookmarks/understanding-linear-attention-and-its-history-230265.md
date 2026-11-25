---
title: "Understanding Linear Attention And Its History"
publishDate: "2026-06-24"
date: "2026-06-24"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "learning-resources"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2069878823018230265-1.jpg"
excerpt: "A video explains linear attention, with a quoted post claiming the 1991 linear Transformer preceded the 2017 Transformer."
---

> Source: [@jbhuang0604 on X](https://x.com/jbhuang0604/status/2069878823018230265) · posted 2026-06-24

![Understanding Linear Attention And Its History (1)](/imgs/blogs/x-bookmarks/2069878823018230265-1.jpg)

![Understanding Linear Attention And Its History (2)](/imgs/blogs/x-bookmarks/2069878823018230265-2.png)

## Summary

@jbhuang0604 says linear attention took a while to click but is satisfying once understood, and links a video explaining its essence and its link to past and modern methods. The quoted post claims Google's 2017 Transformer is based on the principles of the 1991 unnormalized linear Transformer, whose cost scales linearly in input size.

## Background

Standard attention compares every token with every other, so cost grows with the square of length. Linear attention reorganizes the math so cost grows linearly. The priority claim in the quote is the quoted author's view and is disputed in the community.

## What I took away

- Linear attention trades exact attention for linear cost.
- Watch the video for an intuitive route in.
- Treat the history claim as contested.

## Reference

- Original post: [https://x.com/jbhuang0604/status/2069878823018230265](https://x.com/jbhuang0604/status/2069878823018230265)
- Author: [@jbhuang0604](https://x.com/jbhuang0604) on X, posted 2026-06-24
