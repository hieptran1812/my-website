---
title: "Weight Space Programming On Qwen3.8-27B"
publishDate: "2026-11-09"
date: "2026-11-09"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "post-training", "open-models", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2119880941711827025-1.jpg"
excerpt: "A team claims editing nine weight matrices in Qwen3.8-27B reduced thinking and improved scores without training."
---

> Source: [@varien on X](https://x.com/varien/status/2119880941711827025) · posted 2026-11-09

![Weight Space Programming On Qwen3.8-27B (1)](/imgs/blogs/x-bookmarks/2119880941711827025-1.jpg)

![Weight Space Programming On Qwen3.8-27B (2)](/imgs/blogs/x-bookmarks/2119880941711827025-2.jpg)

## Summary

**Qwen Concept 27B** rewrote 9 weight matrices of Qwen3.8-27B with no gradient steps or training data. They claim it thinks less and scores higher: GPQA Diamond +4.3 points, LiveCodeBench v6 +13.0, and 27-54% fewer thinking tokens. Weights and a paper are released, and a quoted post notes an unexpected +13% on coding.

## Background

Weight editing changes parameters directly instead of training. GPQA Diamond and LiveCodeBench are reasoning and coding benchmarks. Fewer thinking tokens means cheaper inference. These are the authors' claims.

## What I took away

- Direct weight edits are a new route to behavior change.
- Cheaper reasoning matters for cost.
- Check robustness results before trusting.

## Reference

- Original post: [https://x.com/varien/status/2119880941711827025](https://x.com/varien/status/2119880941711827025)
- Author: [@varien](https://x.com/varien) on X, posted 2026-11-09
