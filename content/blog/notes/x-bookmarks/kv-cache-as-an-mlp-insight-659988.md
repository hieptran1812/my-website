---
title: "KV Cache as an MLP Insight"
publishDate: "2026-06-10"
date: "2026-06-10"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "kv-cache", "llm-architecture", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2064729744860659988-1.jpg"
excerpt: "A post noting a fixed KV cache acts like an MLP, tied to a quoted result of 200x KV cache compression."
---

> Source: [@part_harry_ on X](https://x.com/part_harry_/status/2064729744860659988) · posted 2026-06-10

![KV Cache as an MLP Insight](/imgs/blogs/x-bookmarks/2064729744860659988-1.jpg)

## Summary

@part_harry_ says a fixed KV cache is an MLP: keys form an up projection, values a down projection, softmax is the nonlinearity. So KV compression is another way of producing weights. The quoted post claims 200x compression in a single forward pass, 36 GiB to about 360 MiB at 256k context.

## Background

An MLP is a feed-forward layer. Seeing attention over fixed memory this way links memory and weights. The 200x is the quoted authors' claim.

## What I took away

- Attention over a cache resembles a small network.
- Compression can be seen as weight generation.

## Reference

- Original post: [https://x.com/part_harry_/status/2064729744860659988](https://x.com/part_harry_/status/2064729744860659988)
- Author: [@part_harry_](https://x.com/part_harry_) on X, posted 2026-06-10
