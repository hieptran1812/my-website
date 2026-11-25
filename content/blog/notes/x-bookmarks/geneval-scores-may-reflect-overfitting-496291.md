---
title: "GenEval Scores May Reflect Overfitting"
publishDate: "2026-06-29"
date: "2026-06-29"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "post-training"]
author: "Hiep Tran"
featured: false
excerpt: "A researcher says SFT can easily reach 90% GenEval, so high scores may signal narrow overfitting."
---

> Source: [@LiangZheng_06 on X](https://x.com/LiangZheng_06/status/2071741332700496291) · posted 2026-06-29

## Summary

@LiangZheng_06 says SFT on data of some distribution easily gets about 90% GenEval, yet models are not truly there. The quoted post says many T2I fine-tuning papers get ~90% and are likely overfitting to narrow SFT data.

## Background

GenEval tests text-to-image models on object and attribute prompts. Fine-tuning on similar prompts can inflate it without real improvement.

## What I took away

- Benchmark gains from fine-tuning can be overfitting.
- Test on out-of-distribution prompts.

## Reference

- Original post: [https://x.com/LiangZheng_06/status/2071741332700496291](https://x.com/LiangZheng_06/status/2071741332700496291)
- Author: [@LiangZheng_06](https://x.com/LiangZheng_06) on X, posted 2026-06-29
