---
title: "Recursive Language Models Reimplemented With Google ADK"
publishDate: "2026-02-07"
date: "2026-02-07"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-agents", "ai-research", "harness-engineering"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2020271355623620801-1.jpg"
excerpt: "A Google Cloud article explains re-implementing the RLM codebase with ADK for handling 10M+ tokens."
---

> Source: [@GoogleCloudTech on X](https://x.com/GoogleCloudTech/status/2020271355623620801) · posted 2026-02-07

![Recursive Language Models Reimplemented With Google ADK](/imgs/blogs/x-bookmarks/2020271355623620801-1.jpg)

## Summary

Google Cloud Tech says Recursive Language Models let agents manage 10M+ tokens by delegating tasks recursively. A Community article explains why ADK was a good choice to re-implement the original RLM codebase in a more enterprise-ready form.

## Background

A Recursive Language Model splits a huge input into pieces and calls itself on sub-parts, so no single call needs the whole context. ADK is Google's Agent Development Kit. This gets around context window limits.

## What I took away

- Recursion is a way to handle inputs larger than the context window.
- Delegation to sub-calls trades latency for scale.
- Read the article for the ADK implementation.

## Reference

- Original post: [https://x.com/GoogleCloudTech/status/2020271355623620801](https://x.com/GoogleCloudTech/status/2020271355623620801)
- Author: [@GoogleCloudTech](https://x.com/GoogleCloudTech) on X, posted 2026-02-07
