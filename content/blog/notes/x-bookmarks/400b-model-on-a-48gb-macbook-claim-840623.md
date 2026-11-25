---
title: "400B Model On A 48GB MacBook Claim"
publishDate: "2026-03-19"
date: "2026-03-19"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "ai-agents", "hardware"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2034666638612840623-1.png"
excerpt: "The author claims a dev ran a roughly 400B model on a 48GB MacBook by giving Claude Code autoresearch and an Apple paper."
---

> Source: [@NainsiDwiv50980 on X](https://x.com/NainsiDwiv50980/status/2034666638612840623) · posted 2026-03-19

![400B Model On A 48GB MacBook Claim](/imgs/blogs/x-bookmarks/2034666638612840623-1.png)

## Summary

The author says someone ran a ~400B parameter model on a laptop with no cloud. The dev fed Claude Code Karpathy's autoresearch repo and Apple's LLM in a Flash paper, with the goal of running Qwen3.5 397B locally on a 48GB MacBook, and it worked. The text is cut off.

## Background

The LLM in a Flash paper is about running models larger than memory by streaming weights from flash storage. The post uses hype language, and speed is not given here. See the original dev's post in another bookmark.

## What I took away

- Weights can be streamed from SSD when RAM is too small.
- Check tokens per second before calling something practical.

## Reference

- Original post: [https://x.com/NainsiDwiv50980/status/2034666638612840623](https://x.com/NainsiDwiv50980/status/2034666638612840623)
- Author: [@NainsiDwiv50980](https://x.com/NainsiDwiv50980) on X, posted 2026-03-19
