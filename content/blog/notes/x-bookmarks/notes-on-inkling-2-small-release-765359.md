---
title: "Notes On Inkling 2 Small Release"
publishDate: "2026-11-07"
date: "2026-11-07"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "moe", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "A hand-written breakdown of Inkling 2 Small, including its 594B-parameter MoE architecture."
---

> Source: [@ahatamiz1 on X](https://x.com/ahatamiz1/status/2119177793317765359) · posted 2026-11-07

## Summary

The author shares notes on the Inkling 2 Small release, stating every word is his own with AI used only for fact checking. He lists a 48-layer decoder-only transformer with 594B total and 17B active parameters and a sparse MoE feed-forward backbone. The post is truncated.

## Background

In MoE models only a subset of parameters (17B here) is used per token while the full set (594B) is stored. Decoder-only means the model generates text left to right.

## What I took away

- Active versus total parameters matter for cost.
- Human-written analyses add useful context.
- See the primary release for exact specs.

## Reference

- Original post: [https://x.com/ahatamiz1/status/2119177793317765359](https://x.com/ahatamiz1/status/2119177793317765359)
- Author: [@ahatamiz1](https://x.com/ahatamiz1) on X, posted 2026-11-07
