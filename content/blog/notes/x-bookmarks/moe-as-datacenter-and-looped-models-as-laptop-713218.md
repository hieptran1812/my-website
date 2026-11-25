---
title: "MoE as Datacenter and Looped Models as Laptop"
publishDate: "2026-11-22"
date: "2026-11-22"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "moe", "llm-architecture", "hardware"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2124292767274713218-1.jpg"
excerpt: "A post argues mixture of experts suits datacenters while looped transformers suit laptops, quoting Inkling 2 Mini."
---

> Source: [@zhengyaojiang on X](https://x.com/zhengyaojiang/status/2124292767274713218) · posted 2026-11-22

![MoE as Datacenter and Looped Models as Laptop](/imgs/blogs/x-bookmarks/2124292767274713218-1.jpg)

## Summary

The author says MoE trades memory for compute with huge weights but few active per token, which works with big batches and big GPUs. On a laptop the batch size is 1 and memory is scarce. The quoted post introduces Inkling 2 Mini, a 26B looped transformer small enough for a laptop with a score of 20.1 on AA Index.

## Background

MoE models route each token to a few experts, so compute is small but all weights must be stored. Looped models reuse layers, so they need less memory. That fits memory-limited local devices.

## What I took away

- Architecture choices depend on deployment hardware.
- MoE needs memory and batching to pay off.
- Looping fits small-batch local use.

## Reference

- Original post: [https://x.com/zhengyaojiang/status/2124292767274713218](https://x.com/zhengyaojiang/status/2124292767274713218)
- Author: [@zhengyaojiang](https://x.com/zhengyaojiang) on X, posted 2026-11-22
