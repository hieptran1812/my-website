---
title: "Training Speculators Is Essential for Production Inference"
publishDate: "2026-07-29"
date: "2026-07-29"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "speculative-decoding", "serving"]
author: "Hiep Tran"
featured: false
excerpt: "Woosuk Kwon says training high-quality speculators is now essential for production inference, welcoming AngelSpec and vLLM support."
---

> Source: [@woosuk_k on X](https://x.com/woosuk_k/status/2082501234171486297) · posted 2026-07-29

## Summary

Woosuk Kwon of vLLM says training high-quality **speculators** has become essential for production-grade inference. He is glad to see the ecosystem growing and vLLM being part of it. The quoted post is Tencent's AngelSpec announcement, claiming about 2x speedup on Hy3-A21B.

## Background

A speculator is the draft model used in speculative decoding. Its quality determines how many tokens get accepted and so how much speedup you get. Serving engines like vLLM need to load and run these drafters efficiently.

## What I took away

- Drafter quality is now a core part of an inference stack.
- Plan for drafter training when deploying a new model.
- Choose drafter tools that work with your serving engine.

## Reference

- Original post: [https://x.com/woosuk_k/status/2082501234171486297](https://x.com/woosuk_k/status/2082501234171486297)
- Author: [@woosuk_k](https://x.com/woosuk_k) on X, posted 2026-07-29
