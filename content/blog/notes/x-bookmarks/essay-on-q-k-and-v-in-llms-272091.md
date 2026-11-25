---
title: "Essay On Q, K And V In LLMs"
publishDate: "2025-11-27"
date: "2025-11-27"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "llm-inference", "learning-resources"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1994024609235272091-1.png"
excerpt: "An engineer shares an essay explaining the query, key and value matrices in language models."
---

> Source: [@arpit_bhayani on X](https://x.com/arpit_bhayani/status/1994024609235272091) · posted 2025-11-27

![Essay On Q, K And V In LLMs](/imgs/blogs/x-bookmarks/1994024609235272091-1.png)

## Summary

The author wrote an essay on what the Q, K and V matrices are and their roles in LLMs. It is a distilled but detailed version that supplements their write-up on how LLM inference works. They say they are no expert.

## Background

In attention, each token produces a query (what it looks for), a key (what it offers) and a value (the content to pass on). Matching queries to keys decides how much of each value is mixed in. Understanding this is the basis for topics like KV caching.

## What I took away

- Query matches keys, and the result weights the values.
- Learn Q, K, V before studying KV caching.
- Pair it with the author's inference write-up.

## Reference

- Original post: [https://x.com/arpit_bhayani/status/1994024609235272091](https://x.com/arpit_bhayani/status/1994024609235272091)
- Author: [@arpit_bhayani](https://x.com/arpit_bhayani) on X, posted 2025-11-27
