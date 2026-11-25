---
title: "DiffusionGemma Tested on Newspaper OCR Correction"
publishDate: "2026-06-11"
date: "2026-06-11"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "open-models", "tools"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2064933218684481918-1.jpg"
excerpt: "A test of DiffusionGemma correcting old newspaper OCR, reported faster and better than an autoregressive baseline."
---

> Source: [@vanstriendaniel on X](https://x.com/vanstriendaniel/status/2064933218684481918) · posted 2026-06-11

![DiffusionGemma Tested on Newspaper OCR Correction](/imgs/blogs/x-bookmarks/2064933218684481918-1.jpg)

## Summary

@vanstriendaniel asks if DiffusionGemma helps fix broken OCR since parallel denoising sees the whole context upfront. On 19th-century newspaper OCR he reports better correction than the autoregressive baseline at about 8x the speed.

## Background

OCR errors are wrong characters from scanned text. A diffusion model can revise all tokens with full context, unlike left-to-right models. Results are one experiment.

## What I took away

- Parallel denoising suits editing tasks.
- One test is not a benchmark.

## Reference

- Original post: [https://x.com/vanstriendaniel/status/2064933218684481918](https://x.com/vanstriendaniel/status/2064933218684481918)
- Author: [@vanstriendaniel](https://x.com/vanstriendaniel) on X, posted 2026-06-11
