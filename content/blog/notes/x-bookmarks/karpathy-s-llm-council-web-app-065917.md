---
title: "Karpathy's LLM Council Web App"
publishDate: "2025-11-24"
date: "2025-11-24"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-agents", "tools", "open-models"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1992822811250065917-1.jpg"
excerpt: "A joke reply to a quoted post where Karpathy builds an llm-council app that sends each query to several models."
---

> Source: [@birdabo on X](https://x.com/birdabo/status/1992822811250065917) · posted 2025-11-24

![Karpathy's LLM Council Web App (1)](/imgs/blogs/x-bookmarks/1992822811250065917-1.jpg)

![Karpathy's LLM Council Web App (2)](/imgs/blogs/x-bookmarks/1992822811250065917-2.jpg)

![Karpathy's LLM Council Web App (3)](/imgs/blogs/x-bookmarks/1992822811250065917-3.jpg)

## Summary

The author jokes that PewDiePie got there first. The quoted post is from Karpathy, who hacked together an **llm-council** web app over a Saturday. It looks like ChatGPT, but each user query is dispatched to several models on a council through OpenRouter, for example openai/gpt-5.1.

## Background

OpenRouter is a service that gives one API to many different LLMs. A council setup asks several models the same question so their answers can be compared or combined. This is a simple way to get a second opinion from different models instead of trusting one.

## What I took away

- Sending one query to several models is easy when you use a routing API like OpenRouter.
- A small weekend project can be a useful way to compare models side by side.
- Disagreement between models can be a signal worth checking.

## Reference

- Original post: [https://x.com/birdabo/status/1992822811250065917](https://x.com/birdabo/status/1992822811250065917)
- Author: [@birdabo](https://x.com/birdabo) on X, posted 2025-11-24
