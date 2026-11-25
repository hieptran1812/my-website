---
title: "CIPHER: Experimental Tiny Architecture on a Laptop"
publishDate: "2026-03-06"
date: "2026-03-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2029796964053962938-1.png"
excerpt: "A hobbyist shares CIPHER, an experimental architecture meant to fix perceived transformer flaws, built with Claude."
---

> Source: [@yashpxl on X](https://x.com/yashpxl/status/2029796964053962938) · posted 2026-03-06

![CIPHER: Experimental Tiny Architecture on a Laptop (1)](/imgs/blogs/x-bookmarks/2029796964053962938-1.png)

![CIPHER: Experimental Tiny Architecture on a Laptop (2)](/imgs/blogs/x-bookmarks/2029796964053962938-2.png)

![CIPHER: Experimental Tiny Architecture on a Laptop (3)](/imgs/blogs/x-bookmarks/2029796964053962938-3.png)

![CIPHER: Experimental Tiny Architecture on a Laptop (4)](/imgs/blogs/x-bookmarks/2029796964053962938-4.png)

## Summary

The author says they built a new architecture called CIPHER from scratch after reading 6 papers and finding 6 flaws in transformers, with help from Claude. The quoted post lists goals: no O(n squared) attention, persistent memory, byte-level input, adaptive compute, sparse activation, and CPU training on an i3 laptop.

## Background

Standard attention cost grows with the square of sequence length. Byte-level models skip tokenization. These are experimental ideas, and the post gives no benchmark results, so quality is unproven.

## What I took away

- Small experiments are a great way to learn architecture ideas.
- Look for benchmarks before accepting claims of fixing transformers.
- LLM assistants can help prototype research ideas.

## Reference

- Original post: [https://x.com/yashpxl/status/2029796964053962938](https://x.com/yashpxl/status/2029796964053962938)
- Author: [@yashpxl](https://x.com/yashpxl) on X, posted 2026-03-06
