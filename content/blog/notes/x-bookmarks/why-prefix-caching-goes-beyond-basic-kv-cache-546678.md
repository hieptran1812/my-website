---
title: "Why Prefix Caching Goes Beyond Basic KV Cache"
publishDate: "2025-11-29"
date: "2025-11-29"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "kv-cache", "serving"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1994745698529546678-1.jpg"
excerpt: "A post argues basic KV cache is not enough when many requests share long prompts, so prefix caching is needed."
---

> Source: [@athletic_coder on X](https://x.com/athletic_coder/status/1994745698529546678) · posted 2025-11-29

![Why Prefix Caching Goes Beyond Basic KV Cache](/imgs/blogs/x-bookmarks/1994745698529546678-1.jpg)

## Summary

The author mocks the advice to just use KV cache. They list cases where it falls short: the same 5000-token system prompt on every request, 10M+ daily requests burning $50k on redundant compute, and RAG pipelines reprocessing the same documents many times. In those cases they say you implement prefix caching.

## Background

A KV cache stores attention keys and values for tokens already processed, so each new token does not recompute them within one request. Prefix caching extends this by reusing the cache across requests that start with the same text. This saves compute when prompts share long common prefixes.

## What I took away

- Put stable content like system prompts at the start so it can be cached.
- Prefix caching helps most with shared prompts and repeated documents.
- Cost savings depend on how much your traffic actually repeats.

## Reference

- Original post: [https://x.com/athletic_coder/status/1994745698529546678](https://x.com/athletic_coder/status/1994745698529546678)
- Author: [@athletic_coder](https://x.com/athletic_coder) on X, posted 2025-11-29
