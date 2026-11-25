---
title: "Prompt Caching As A Cost Saver"
publishDate: "2025-12-19"
date: "2025-12-19"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "kv-cache", "serving"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2002016000447828469-1.jpg"
excerpt: "Elvis Saravia praises a post on prompt caching as an underused way to cut LLM costs."
---

> Source: [@omarsar0 on X](https://x.com/omarsar0/status/2002016000447828469) · posted 2025-12-19

![Prompt Caching As A Cost Saver](/imgs/blogs/x-bookmarks/2002016000447828469-1.jpg)

## Summary

The author calls a post on prompt caching brilliant. He says prompt caching is one of the most effective and underused techniques for lowering LLM usage costs.

## Background

Prompt caching reuses the work done on a repeated prompt prefix, such as a long system prompt, so it does not need to be processed again. Providers often charge much less for cached tokens and respond faster. It works best when the stable part comes first in the prompt.

## What I took away

- Put stable content like system prompts at the start so it can be cached.
- Prompt caching can cut both cost and latency.
- Check your provider's caching rules and pricing.

## Reference

- Original post: [https://x.com/omarsar0/status/2002016000447828469](https://x.com/omarsar0/status/2002016000447828469)
- Author: [@omarsar0](https://x.com/omarsar0) on X, posted 2025-12-19
