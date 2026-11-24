---
title: "Tencent Open Sources AngelSpec Speculative Decoding Framework"
publishDate: "2026-07-29"
date: "2026-07-29"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "speculative-decoding", "llm-inference", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "Tencent Hunyuan releases AngelSpec for training and deploying speculative decoders, claiming about 2x end-to-end speedup on Hy3-A21B."
---

> Source: [@TencentHunyuan on X](https://x.com/TencentHunyuan/status/2082447023626944936) · posted 2026-07-29

## Summary

Tencent Hunyuan says it open sourced **AngelSpec**, an end-to-end speculative decoding framework for both training and deployment. On Hy3-A21B, its DFly method is claimed to give a 1.98 to 2.40x end-to-end speedup over normal autoregressive decoding at concurrency from 4 to 64, plus 10.5 to 11.8% higher results on another metric (text cut off). The GitHub page calls it a torch-native training framework for MTP and block-parallel drafting.

## Background

Autoregressive decoding generates one token per step. Speculative decoding drafts several tokens and verifies them in one big-model pass. Speedups usually shrink as concurrency rises because the GPU is already busy, so reporting several concurrency levels is useful.

## What I took away

- Check speculative decoding speedups at your real concurrency level.
- Training your own drafter is now supported by several open frameworks.
- Treat the speedup range as the authors' measured claim on their model.

## Reference

- Original post: [https://x.com/TencentHunyuan/status/2082447023626944936](https://x.com/TencentHunyuan/status/2082447023626944936)
- Author: [@TencentHunyuan](https://x.com/TencentHunyuan) on X, posted 2026-07-29
