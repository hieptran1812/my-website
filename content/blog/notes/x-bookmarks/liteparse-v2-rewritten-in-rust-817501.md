---
title: "LiteParse v2 Rewritten in Rust"
publishDate: "2026-05-27"
date: "2026-05-27"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "tools", "ai-agents"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2059710330016817501-1.png"
excerpt: "Jerry Liu announces LiteParse v2, claiming the fastest and most accurate model-free open source PDF parser."
---

> Source: [@jerryjliu0 on X](https://x.com/jerryjliu0/status/2059710330016817501) · posted 2026-05-27

![LiteParse v2 Rewritten in Rust](/imgs/blogs/x-bookmarks/2059710330016817501-1.png)

## Summary

Jerry Liu says they created the world's fastest PDF parser, and that it is more accurate than other open-source, model-free parsers like pymupdf, pypdf, markitdown and pdftotext. The quoted release notes say v2.0 was rewritten in Rust, up to 100x faster, installable in Rust, JS/TS and Python, with a WASM package for browsers and edge.

## Background

Model-free means no neural network is used to read the document, so parsing is fast and cheap. The speed and accuracy claims are from the vendor.

## What I took away

- Up to 100x is the vendor's best case; test on your files.
- Multi-language bindings make it easy to adopt.
- A fast parser can remove a bottleneck in document pipelines.

## Reference

- Original post: [https://x.com/jerryjliu0/status/2059710330016817501](https://x.com/jerryjliu0/status/2059710330016817501)
- Author: [@jerryjliu0](https://x.com/jerryjliu0) on X, posted 2026-05-27
