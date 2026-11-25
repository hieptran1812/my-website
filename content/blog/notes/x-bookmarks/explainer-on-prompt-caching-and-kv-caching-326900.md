---
title: "Explainer On Prompt Caching And KV Caching"
publishDate: "2025-12-20"
date: "2025-12-20"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "kv-cache", "llm-inference", "learning-resources"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2002419594519326900-1.jpg"
excerpt: "A post recommends an ngrok blog explaining how LLMs work and how prompt caching makes tokens 10x cheaper."
---

> Source: [@joshelman on X](https://x.com/joshelman/status/2002419594519326900) · posted 2025-12-20

![Explainer On Prompt Caching And KV Caching](/imgs/blogs/x-bookmarks/2002419594519326900-1.jpg)

## Summary

The author calls the linked piece one of the best explainers he has read on how LLMs work overall, and says it also explains KV caching. The link card is titled Prompt caching: 10x cheaper LLM tokens, but how? from the ngrok blog.

## Background

The KV cache stores attention keys and values for tokens already processed. Prompt caching reuses that stored work when a new request starts with the same prefix, so you do not pay to recompute it. This is why stable prompt prefixes cost less.

## What I took away

- Put stable content at the start of prompts to benefit from caching.
- The KV cache is what makes caching of prompts possible.
- Read the ngrok post for a clear walkthrough.

## Reference

- Original post: [https://x.com/joshelman/status/2002419594519326900](https://x.com/joshelman/status/2002419594519326900)
- Author: [@joshelman](https://x.com/joshelman) on X, posted 2025-12-20
