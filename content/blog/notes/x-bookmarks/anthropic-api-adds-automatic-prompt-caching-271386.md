---
title: "Anthropic API Adds Automatic Prompt Caching"
publishDate: "2026-02-19"
date: "2026-02-19"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "kv-cache", "llm-inference", "tools"]
author: "Hiep Tran"
featured: false
excerpt: "Alex Albert announces automatic prompt caching in the Anthropic API with no cache points needed."
---

> Source: [@alexalbert__ on X](https://x.com/alexalbert__/status/2024586006633271386) · posted 2026-02-19

## Summary

The post says automatic prompt caching was added to the API, so developers no longer set cache points in requests.

## Background

Prompt caching reuses the processed form of repeated prompt prefixes, lowering cost and latency. Before, developers marked where to cache.

## What I took away

- Cache repeated prefixes.
- Keep stable content at the start of prompts.

## Reference

- Original post: [https://x.com/alexalbert__/status/2024586006633271386](https://x.com/alexalbert__/status/2024586006633271386)
- Author: [@alexalbert__](https://x.com/alexalbert__) on X, posted 2026-02-19
