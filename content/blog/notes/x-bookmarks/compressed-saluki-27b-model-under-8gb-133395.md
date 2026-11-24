---
title: "Compressed Saluki 27B Model Under 8GB"
publishDate: "2026-10-08"
date: "2026-10-08"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "quantization", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "Underdog released Saluki 27B, a compressed Qwen 3.8 27B variant claimed to keep 96% of benchmark performance."
---

> Source: [@loyal on X](https://x.com/loyal/status/2108021482983133395) · posted 2026-10-08

## Summary

The author announces **Underdog Saluki 27B**, based on Qwen 3.8 27B. They claim it is almost 7x smaller than the full-precision model, keeps 96% of benchmark performance, and beats Qwen at tool calling. They call it the best model under 8GB; the text is cut off after that.

## Background

Model compression, usually through quantization, stores weights with fewer bits so a large model fits in less memory. The trade-off is some loss in quality, which the 96% figure tries to measure.

## What I took away

- A well-compressed 27B model can fit consumer hardware.
- Check which benchmarks the 96% figure covers before relying on it.
- Tool-calling quality is worth testing separately after compression.

## Reference

- Original post: [https://x.com/loyal/status/2108021482983133395](https://x.com/loyal/status/2108021482983133395)
- Author: [@loyal](https://x.com/loyal) on X, posted 2026-10-08
