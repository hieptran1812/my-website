---
title: "Video Delta Net Speeds Up MiniMax H3 Video Generation"
publishDate: "2026-09-02"
date: "2026-09-02"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "audio-video", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "Video Delta Net uses hybrid attention to make MiniMax H3 video generation 75 to 90 times faster, near real time."
---

> Source: [@xiuyu_l on X](https://x.com/xiuyu_l/status/2095284110264766874) · posted 2026-09-02

## Summary

The author says their team made **MiniMax H3** 75 to 90 times faster, generating a 14-second video in 11 seconds. The quoted announcement introduces **Video Delta Net (VDN)**, a hybrid attention method for live text-to-video, claimed to keep near-lossless quality while producing 14 seconds of 768p video in about 11 seconds.

## Background

Video models attend over huge numbers of tokens across space and time, and full attention cost grows with the square of that length. Hybrid attention mixes cheaper attention types with some full attention to cut that cost. Generating faster than playback means video could be produced live.

## What I took away

- Attention cost is the main bottleneck in video generation, so attention changes give the largest speedups.
- Faster-than-playback generation opens up live and interactive video uses.
- The near-lossless quality claim is the authors'; compare sample outputs yourself.

## Reference

- Original post: [https://x.com/xiuyu_l/status/2095284110264766874](https://x.com/xiuyu_l/status/2095284110264766874)
- Author: [@xiuyu_l](https://x.com/xiuyu_l) on X, posted 2026-09-02
