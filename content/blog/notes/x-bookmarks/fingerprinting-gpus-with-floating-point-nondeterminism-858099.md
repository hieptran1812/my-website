---
title: "Fingerprinting GPUs With Floating-Point Nondeterminism"
publishDate: "2026-10-31"
date: "2026-10-31"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "hardware"]
author: "Hiep Tran"
featured: false
excerpt: "A write-up shows a default torch.index_add_ call can fingerprint an exact GPU, even among identical models."
---

> Source: [@cHHillee on X](https://x.com/cHHillee/status/2116662918091858099) · posted 2026-10-31

## Summary

The quoted write-up shows you can fingerprint the exact GPU you are running on with a default torch.index_add_ call, even among identical GPUs. It works because floating-point math depends on operation order and each GPU has its own, and the fingerprint survives reboots and cannot be faked. @cHHillee says it is a new and cool thing to learn about GPU nondeterminism.

## Background

Floating-point addition is not associative: (a+b)+c can differ slightly from a+(b+c). Operations like index_add_ use atomic adds whose order depends on hardware timing, so results vary in tiny, device-specific ways.

## What I took away

- GPU results can differ slightly run to run because of operation order.
- Use deterministic modes when you need reproducible results.
- Hardware fingerprints have privacy and security implications.

## Reference

- Original post: [https://x.com/cHHillee/status/2116662918091858099](https://x.com/cHHillee/status/2116662918091858099)
- Author: [@cHHillee](https://x.com/cHHillee) on X, posted 2026-10-31
