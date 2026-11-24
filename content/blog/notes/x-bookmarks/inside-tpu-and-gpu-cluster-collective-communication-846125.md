---
title: "Inside TPU and GPU Cluster Collective Communication"
publishDate: "2026-07-14"
date: "2026-07-14"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "hardware", "learning-resources", "moe"]
author: "Hiep Tran"
featured: false
excerpt: "Aleksa Gordic publishes an in-depth blog post on collective communication primitives behind scaling training and inference."
---

> Source: [@gordic_aleksa on X](https://x.com/gordic_aleksa/status/2077075070233846125) · posted 2026-07-14

## Summary

The author announces a blog post titled Inside TPU and GPU Clusters: The Anatomy of Collective Communication. It aims to help readers understand the core primitives behind scaling training and inference for MoE and dense transformers, one level below FSDP, expert parallelism and data parallelism.

## Background

Collective communication covers operations like all-reduce, all-gather and all-to-all, where many chips exchange data. Parallelism strategies such as FSDP (sharding model state across GPUs) and expert parallelism (spreading MoE experts across GPUs) are built on these operations, so their cost determines scaling efficiency.

## What I took away

- Understanding all-reduce and all-to-all explains most distributed training bottlenecks.
- MoE models depend heavily on all-to-all communication.
- Save this as a reference for distributed systems learning.

## Reference

- Original post: [https://x.com/gordic_aleksa/status/2077075070233846125](https://x.com/gordic_aleksa/status/2077075070233846125)
- Author: [@gordic_aleksa](https://x.com/gordic_aleksa) on X, posted 2026-07-14
