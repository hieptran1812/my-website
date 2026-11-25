---
title: "Live Avatar Keeps Characters Consistent In Long Videos"
publishDate: "2025-12-07"
date: "2025-12-07"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "audio-video", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1997500053515358335-1.jpg"
excerpt: "Alibaba's Live Avatar work updates the sink frame's RoPE to keep characters consistent in videos longer than 8 seconds."
---

> Source: [@andrew_n_carr on X](https://x.com/andrew_n_carr/status/1997500053515358335) · posted 2025-12-07

![Live Avatar Keeps Characters Consistent In Long Videos](/imgs/blogs/x-bookmarks/1997500053515358335-1.jpg)

## Summary

@andrew_n_carr says a major challenge in video generation is keeping characters consistent for videos longer than 8 seconds. They say Alibaba's Live Avatar work proposes a perfect solution by dynamically updating the sink frame's RoPE to preserve relative positioning. The text is cut off after that.

## Background

RoPE (rotary position embedding) tells a model where each token or frame sits in the sequence. A sink frame is a reference frame kept in attention. Updating its position as the video grows helps the model keep the identity from drifting. 'Perfect solution' is the author's wording.

## What I took away

- Position encoding choices affect long-video consistency.
- Keeping a fixed reference frame helps preserve identity.

## Reference

- Original post: [https://x.com/andrew_n_carr/status/1997500053515358335](https://x.com/andrew_n_carr/status/1997500053515358335)
- Author: [@andrew_n_carr](https://x.com/andrew_n_carr) on X, posted 2025-12-07
