---
title: "LiteParse v2 Rust PDF Parser Speed Claim"
publishDate: "2026-05-29"
date: "2026-05-29"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "tools", "open-models", "ai-agents"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2060401682610262424-1.jpg"
excerpt: "Jerry Liu shares LiteParse v2, a Rust rewrite he claims is the fastest and most accurate model-free open source PDF parser."
---

> Source: [@jerryjliu0 on X](https://x.com/jerryjliu0/status/2060401682610262424) · posted 2026-05-29

![LiteParse v2 Rust PDF Parser Speed Claim (1)](/imgs/blogs/x-bookmarks/2060401682610262424-1.jpg)

![LiteParse v2 Rust PDF Parser Speed Claim (2)](/imgs/blogs/x-bookmarks/2060401682610262424-2.png)

## Summary

Jerry Liu comments that parsing PDFs is extremely fast, with a video shown at 1x. The quoted post claims LiteParse v2 is the world's fastest PDF parser and more accurate than other open-source, model-free parsers such as pymupdf, pypdf and markitdown. It was rewritten in Rust.

## Background

A model-free parser extracts text using rules and layout analysis, not a neural network. That makes it fast and cheap. Speed and accuracy claims come from the authors and should be checked on your own documents.

## What I took away

- Model-free parsers are a cheap first step before using a VLM for PDFs.
- Benchmark parsers on your own document types.
- Rust rewrites often bring big speedups for I/O-heavy tools.

## Reference

- Original post: [https://x.com/jerryjliu0/status/2060401682610262424](https://x.com/jerryjliu0/status/2060401682610262424)
- Author: [@jerryjliu0](https://x.com/jerryjliu0) on X, posted 2026-05-29
