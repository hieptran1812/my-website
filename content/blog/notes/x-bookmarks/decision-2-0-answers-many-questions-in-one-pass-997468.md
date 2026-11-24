---
title: "Decision 2.0 Answers Many Questions in One Pass"
publishDate: "2026-10-09"
date: "2026-10-09"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "open-models", "serving"]
author: "Hiep Tran"
featured: false
excerpt: "vLLM highlights Decision 2.0, open models that answer several questions about one input in a single forward pass."
---

> Source: [@vllm_project on X](https://x.com/vllm_project/status/2108441335044997468) · posted 2026-10-09

## Summary

vLLM praises the Semantic Router team for Decision 2.0, which answers multiple questions about one input in a single forward pass with per-option probabilities. They say it is useful for routing and classification. The quoted post says it includes six open models from 0.6B to 27B, each topping same-size open models on the Jev Decision Index 0.3 in their comparison.

## Background

A forward pass is one run of the model over the input. Getting several classification answers from one pass, with probabilities, is much cheaper than calling the model once per question.

## What I took away

- Batching several classification questions into one pass saves cost in routers.
- Per-option probabilities let you set confidence thresholds.

## Reference

- Original post: [https://x.com/vllm_project/status/2108441335044997468](https://x.com/vllm_project/status/2108441335044997468)
- Author: [@vllm_project](https://x.com/vllm_project) on X, posted 2026-10-09
