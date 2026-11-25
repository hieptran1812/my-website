---
title: "Baidu Open-Sources Unlimited OCR With R-SWA"
publishDate: "2026-06-23"
date: "2026-06-23"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "open-models", "moe", "llm-architecture"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2069358973753729165-1.jpg"
excerpt: "Baidu open-sources Unlimited OCR, a 3B MoE model with Reference Sliding Window Attention for long documents."
---

> Source: [@Baidu_Inc on X](https://x.com/Baidu_Inc/status/2069358973753729165) · posted 2026-06-23

![Baidu Open-Sources Unlimited OCR With R-SWA](/imgs/blogs/x-bookmarks/2069358973753729165-1.jpg)

## Summary

@Baidu_Inc announces Unlimited OCR, with 3B total and 500M activated parameters, able to transcribe 40+ pages in one pass. The quoted post says it sets new end-to-end SOTA results on OmniDocBench v1.5 and v1.6, and names Reference Sliding Window Attention (R-SWA) as the key innovation.

## Background

OCR models convert document images to text. Sliding window attention limits each token to nearby tokens, keeping memory bounded for long inputs. The SOTA claim comes from the company.

## What I took away

- R-SWA is the technique to look up for long-document reading.
- Benchmark claims are the vendor's own.

## Reference

- Original post: [https://x.com/Baidu_Inc/status/2069358973753729165](https://x.com/Baidu_Inc/status/2069358973753729165)
- Author: [@Baidu_Inc](https://x.com/Baidu_Inc) on X, posted 2026-06-23
