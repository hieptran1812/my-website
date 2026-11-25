---
title: "Running Inkling 2 Mini Fast on Local Macs"
publishDate: "2026-11-21"
date: "2026-11-21"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "open-models", "hardware"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2124028580573450484-1.jpg"
excerpt: "A post shares optimization work making Inkling 2 Mini run fast on Apple hardware, with decode and prefill speeds."
---

> Source: [@0xSero on X](https://x.com/0xSero/status/2124028580573450484) · posted 2026-11-21

![Running Inkling 2 Mini Fast on Local Macs](/imgs/blogs/x-bookmarks/2124028580573450484-1.jpg)

## Summary

The author says they worked on optimizing Inkling-2-small to give Sonnet-level performance on a laptop. The quoted post says that with Thinking Machines, Inkling 2 Mini ran at 210 tok/s decode and 3620 tok/s prefill on a 96GB M5 Ultra Mac Studio and 125 tok/s decode on a 48GB M5 Max MacBook Pro, and built a weather app 1.8x faster than Claude Sonnet 5.5.

## Background

Prefill processes the prompt and decode generates new tokens one by one. Faster local speeds make laptop agents practical. The quality comparison is the author's claim.

## What I took away

- Local decode speeds above 100 tok/s are reported on laptops.
- Compare prefill and decode separately.
- Performance claims need independent checks.

## Reference

- Original post: [https://x.com/0xSero/status/2124028580573450484](https://x.com/0xSero/status/2124028580573450484)
- Author: [@0xSero](https://x.com/0xSero) on X, posted 2026-11-21
