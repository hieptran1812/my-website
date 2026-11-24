---
title: "Sol Attention Sparse Attention for Video Diffusion"
publishDate: "2026-07-21"
date: "2026-07-21"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "audio-video", "gpu-kernels"]
author: "Hiep Tran"
featured: false
excerpt: "Song Han shares Sol-Engine's update featuring Sol Attention, a training-free sparse attention method for video diffusion."
---

> Source: [@songhan_mit on X](https://x.com/songhan_mit/status/2079597618376692130) · posted 2026-07-21

## Summary

Song Han describes the work as speed-of-light block sparse attention. The quoted Sol-Engine weekly update says they integrated **Sol Attention**, a new training-free sparse attention method for video diffusion. The full paper is coming next week and the method is already available in the engine.

## Background

Video diffusion models attend over huge numbers of tokens across frames, so full attention is very expensive. Block sparse attention computes only the important blocks of the attention matrix. Training-free means it can be applied to existing models without retraining.

## What I took away

- Sparse attention is a main lever for faster video generation.
- Training-free methods are easy to try on existing models.
- Wait for the paper to judge quality loss versus speed.

## Reference

- Original post: [https://x.com/songhan_mit/status/2079597618376692130](https://x.com/songhan_mit/status/2079597618376692130)
- Author: [@songhan_mit](https://x.com/songhan_mit) on X, posted 2026-07-21
