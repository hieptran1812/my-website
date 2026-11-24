---
title: "Two Teams Reach Similar MiniMax H3 Video Speedups"
publishDate: "2026-08-28"
date: "2026-08-28"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "audio-video", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "The author notes FastH3 and fal's Minimax H3 Max independently reached about the same real-time video speedup."
---

> Source: [@aisearchio on X](https://x.com/aisearchio/status/2093436718615822660) · posted 2026-08-28

## Summary

The author calls it a coincidence that FastH3 reached almost the same speedup as **Minimax H3 Max** from fal, which was done independently and whose weights will be released soon. The quoted post says a Minimax H3 finetune can generate 15 seconds of video in 13 seconds, up to 14x faster, and is fully open with training code coming soon.

## Background

When separate teams converge on similar speedups, it often means they hit a common limit or used similar techniques such as step distillation and sparse attention. Real time here means generation takes about as long as the video plays.

## What I took away

- Independent teams reaching similar numbers suggests a shared practical ceiling for now.
- Open weights and training code allow direct comparison between approaches.
- The field moves fast, so check dates when comparing results.

## Reference

- Original post: [https://x.com/aisearchio/status/2093436718615822660](https://x.com/aisearchio/status/2093436718615822660)
- Author: [@aisearchio](https://x.com/aisearchio) on X, posted 2026-08-28
