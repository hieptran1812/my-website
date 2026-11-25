---
title: "GLM-5.2 4-bit With BF16 MTP Head"
publishDate: "2026-06-29"
date: "2026-06-29"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "quantization", "speculative-decoding", "serving"]
author: "Hiep Tran"
featured: false
excerpt: "A post reports a 4-bit GLM-5.2 keeping its MTP head in BF16, matching FP8 quality on fewer GPUs."
---

> Source: [@canadaquant on X](https://x.com/canadaquant/status/2071435463928525108) · posted 2026-06-29

## Summary

@canadaquant says they quantized GLM-5.2 (744B MoE) to 4-bit, kept MTP draft head in BF16, matched FP8 quality, run on 4xH200 instead of 8, and claim +69-79% vs AWQ/NVFP4 at batch-1.

## Background

MTP is multi-token prediction used for speculative decoding. Keeping the draft head precise preserves acceptance.

## What I took away

- Keep sensitive parts in higher precision.
- Quantization can halve GPU needs.

## Reference

- Original post: [https://x.com/canadaquant/status/2071435463928525108](https://x.com/canadaquant/status/2071435463928525108)
- Author: [@canadaquant](https://x.com/canadaquant) on X, posted 2026-06-29
