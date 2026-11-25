---
title: "GLM 5.2 Self-Hosted On 8x H200"
publishDate: "2026-07-03"
date: "2026-07-03"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "open-models", "serving"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2073072210915598579-1.jpg"
excerpt: "A user reports GLM 5.2 running near 100 tokens per second on 8 H200 GPUs."
---

> Source: [@AtharvaIngle7 on X](https://x.com/AtharvaIngle7/status/2073072210915598579) · posted 2026-07-03

![GLM 5.2 Self-Hosted On 8x H200](/imgs/blogs/x-bookmarks/2073072210915598579-1.jpg)

## Summary

@AtharvaIngle7 reports running GLM 5.2 in pi at almost 100 tokens per second on 8x H200, and calls it a frontier-level model that can be self-hosted. Next steps are testing quants and finding the minimum GPU setup.

## Background

Tokens per second measures generation speed. Quantization stores weights in fewer bits to cut memory. Finding a minimal GPU setup lowers the cost of self-hosting.

## What I took away

- Big open models can be served on a single 8-GPU node.
- Quantization is the next lever for lowering hardware needs.
- Measure speed on your own workload.

## Reference

- Original post: [https://x.com/AtharvaIngle7/status/2073072210915598579](https://x.com/AtharvaIngle7/status/2073072210915598579)
- Author: [@AtharvaIngle7](https://x.com/AtharvaIngle7) on X, posted 2026-07-03
