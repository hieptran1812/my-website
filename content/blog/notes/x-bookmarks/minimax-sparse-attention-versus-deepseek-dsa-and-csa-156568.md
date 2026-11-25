---
title: "MiniMax Sparse Attention Versus DeepSeek DSA and CSA"
publishDate: "2026-05-26"
date: "2026-05-26"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "kv-cache", "llm-inference"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2059321928205156568-1.jpg"
excerpt: "Elie Bakouch compares MiniMax's new sparse attention with DeepSeek V3.2 DSA and V4 CSA."
---

> Source: [@eliebakouch on X](https://x.com/eliebakouch/status/2059321928205156568) · posted 2026-05-26

![MiniMax Sparse Attention Versus DeepSeek DSA and CSA (1)](/imgs/blogs/x-bookmarks/2059321928205156568-1.jpg)

![MiniMax Sparse Attention Versus DeepSeek DSA and CSA (2)](/imgs/blogs/x-bookmarks/2059321928205156568-2.jpg)

## Summary

The author compares the new MiniMax sparse attention to DeepSeek V3.2 (DSA) and V4 (CSA). Main changes: it is based on GQA, not MLA, and uses block-level selection like CSA, but attention is done on the real KV instead of in the compressed dimension. The quote hints something big is coming.

## Background

Sparse attention lets each token look at only some past tokens, saving compute on long contexts. GQA shares key/value heads, and MLA compresses them. Where attention runs matters for quality and memory.

## What I took away

- Sparse attention designs differ in what they select and where they compute.
- Attention on real KV avoids compression error but costs more memory.
- Compare these designs when thinking about long-context efficiency.

## Reference

- Original post: [https://x.com/eliebakouch/status/2059321928205156568](https://x.com/eliebakouch/status/2059321928205156568)
- Author: [@eliebakouch](https://x.com/eliebakouch) on X, posted 2026-05-26
