---
title: "OpenAI Habitat Storage Platform Rewritten in Rust"
publishDate: "2026-09-11"
date: "2026-09-11"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "industry-news", "tools"]
author: "Hiep Tran"
featured: false
excerpt: "OpenAI describes Habitat, the storage platform behind ChatGPT and Codex, and its rewrite from Python to Rust."
---

> Source: [@jxnlco on X](https://x.com/jxnlco/status/2098503053594841163) · posted 2026-09-11

## Summary

The quoted post says **Habitat** is OpenAI's online storage platform powering ChatGPT and Codex, growing over 10x year over year. Before a Rust rewrite, its Python service handled over 20 million requests per second at peak. The author is glad the write-up was published.

## Background

Online storage serves data with low latency for live applications. Python is productive but has runtime overhead; Rust gives speed and memory safety. Rewrites like this are common when a service hits scale limits.

## What I took away

- Python can go very far before a rewrite is needed.
- Read the write-up for lessons on migrating a high-traffic service.
- Rewrite to a faster language when growth makes overhead costly.

## Reference

- Original post: [https://x.com/jxnlco/status/2098503053594841163](https://x.com/jxnlco/status/2098503053594841163)
- Author: [@jxnlco](https://x.com/jxnlco) on X, posted 2026-09-11
