---
title: "Matryoshka LM Suites Train Nested Models in One Run"
publishDate: "2026-08-19"
date: "2026-08-19"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "llm-architecture", "distillation"]
author: "Hiep Tran"
featured: false
excerpt: "A paper on training a whole suite of language model sizes nested inside one architecture in a single run."
---

> Source: [@COLM on X](https://x.com/nthngdy/status/2090073045146677693) · posted 2026-08-19

## Summary

The thread announces a paper on Matryoshka LM Suites. Instead of training each model size in a suite separately, the models are nested within a single architecture and trained together in one run. The authors claim this uses much less compute with the same performance, and also enables free distillation and better speculative decoding.

## Background

A model suite is a family of the same model at several sizes, like small, medium and large. Matryoshka refers to Russian nesting dolls: smaller models live inside the larger one and share weights. Nested models naturally make good draft models for speculative decoding because they behave similarly to the big one.

## What I took away

- Sharing weights across model sizes can cut the cost of training a family of models.
- Nested small models are natural candidates for drafters in speculative decoding.
- Check the paper for how the same performance claim was measured.

## Reference

- Original post: [https://x.com/nthngdy/status/2090073045146677693](https://x.com/nthngdy/status/2090073045146677693)
- Author: [@COLM](https://x.com/COLM) on X, posted 2026-08-19
