---
title: "Dolphin Tool for Parsing Document Images and PDFs"
publishDate: "2026-01-09"
date: "2026-01-09"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "tools", "open-models", "llm-inference"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2009751740925763898-1.jpg"
excerpt: "A tool that turns document images and PDFs into structured Markdown or JSON."
---

> Source: [@TheTuringPost on X](https://x.com/TheTuringPost/status/2009751740925763898) · posted 2026-01-09

![Dolphin Tool for Parsing Document Images and PDFs](/imgs/blogs/x-bookmarks/2009751740925763898-1.jpg)

## Summary

The post recommends Dolphin for getting structured Markdown or JSON from document images and PDFs. It handles multi-page PDFs and works with vLLM, TensorRT-LLM and Hugging Face. It detects scanned versus digital documents, recovers layout and reading order, and parses text, tables and formulas.

## Background

Document parsing converts messy PDFs and scans into clean text with structure. This is a common first step for RAG and data pipelines. vLLM and TensorRT-LLM are inference engines for running models fast.

## What I took away

- Good parsing matters before any LLM can use a PDF.
- Preserving layout and reading order improves downstream quality.
- Try it on your own hard documents before adopting it.

## Reference

- Original post: [https://x.com/TheTuringPost/status/2009751740925763898](https://x.com/TheTuringPost/status/2009751740925763898)
- Author: [@TheTuringPost](https://x.com/TheTuringPost) on X, posted 2026-01-09
