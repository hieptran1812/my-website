---
title: "qwenFS Filesystem Compiled Into Model Weights"
publishDate: "2026-11-04"
date: "2026-11-04"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "interpretability"]
author: "Hiep Tran"
featured: false
excerpt: "A project adds a filesystem directly into Qwen3.5-2B's weights, following earlier CPU and GPU builds inside the model."
---

> Source: [@contextkingceo on X](https://x.com/contextkingceo/status/2117831173468737610) · posted 2026-11-04

## Summary

The quoted post introduces qwenFS: Qwen3.5-2B with a filesystem compiled directly into its weights, with nothing running outside the model. It adds 8 layers in front of Qwen's own 24, using attention to hold files and Gated DeltaNet layers as well. It follows earlier work putting a CPU and a GPU into the model. @contextkingceo says he learns a lot from the author every day.

## Background

Compiling a program into weights means setting weights by hand so the network computes a known algorithm, instead of learning it from data. Attention can act like a lookup table over stored items. Gated DeltaNet is a linear-attention style layer with a recurrent state that can act like memory.

## What I took away

- Transformers can implement exact algorithms when weights are set by hand.
- Attention can be viewed as content-addressable memory.
- These builds are useful for understanding what layers can compute.

## Reference

- Original post: [https://x.com/contextkingceo/status/2117831173468737610](https://x.com/contextkingceo/status/2117831173468737610)
- Author: [@contextkingceo](https://x.com/contextkingceo) on X, posted 2026-11-04
