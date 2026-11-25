---
title: "GLM 5.2 4-bit On M3 Ultra"
publishDate: "2026-07-04"
date: "2026-07-04"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "quantization", "hardware"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2073502277449486460-1.jpg"
excerpt: "A post shows GLM 5.2 at 4-bit running at about 16 tokens per second on a 512GB M3 Ultra."
---

> Source: [@ivanfioravanti on X](https://x.com/ivanfioravanti/status/2073502277449486460) · posted 2026-07-04

![GLM 5.2 4-bit On M3 Ultra (1)](/imgs/blogs/x-bookmarks/2073502277449486460-1.jpg)

![GLM 5.2 4-bit On M3 Ultra (2)](/imgs/blogs/x-bookmarks/2073502277449486460-2.jpg)

![GLM 5.2 4-bit On M3 Ultra (3)](/imgs/blogs/x-bookmarks/2073502277449486460-3.jpg)

## Summary

@ivanfioravanti shows GLM 5.2 4bit on a single M3 Ultra 512GB at around 16 toks/s, versus about 17 toks/s for a q2 version in a ds4-eval video. The quoted post covers DawrfStar results with DeepSeek 4 Flash, PRO and GLM 5.2 on hard programming tasks judged by GPT 5.5, and says GLM 5.2 support will extend to CUDA and Strix Halo.

## Background

Unified memory machines like the M3 Ultra can hold very large quantized models. A 4-bit model uses about half the memory of 8-bit but may lose some quality.

## What I took away

- Large quantized MoE models can run on one high-memory desktop.
- Lower bit counts give small speed differences here.
- Judge quality on your own tasks.

## Reference

- Original post: [https://x.com/ivanfioravanti/status/2073502277449486460](https://x.com/ivanfioravanti/status/2073502277449486460)
- Author: [@ivanfioravanti](https://x.com/ivanfioravanti) on X, posted 2026-07-04
