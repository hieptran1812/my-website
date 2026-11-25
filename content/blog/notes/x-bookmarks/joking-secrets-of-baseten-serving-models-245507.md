---
title: "Joking Secrets Of Baseten Serving Models"
publishDate: "2026-11-09"
date: "2026-11-09"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "llm-inference", "industry-news"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2119841165491245507-1.jpg"
excerpt: "A tongue-in-cheek post lists supposed secrets behind Baseten running 9B and 27B models."
---

> Source: [@feilsystem on X](https://x.com/feilsystem/status/2119841165491245507) · posted 2026-11-09

![Joking Secrets Of Baseten Serving Models](/imgs/blogs/x-bookmarks/2119841165491245507-1.jpg)

## Summary

The author jokes that Baseten runs 9B and 27B models by training them to need one inference pass per state, not a prefill step per question. A second joke says they asked codex to make inference faster and make no mistakes. The post mentions perplexity flash v1.2 could be 30x more expensive.

## Background

Prefill is the step where the model processes the whole prompt before generating. Avoiding repeated prefill saves compute. The tone is humorous, so treat the claims as satire.

## What I took away

- Repeated prefill is a major cost driver.
- Do not read joke posts as technical specs.
- Design outputs to reuse computation.

## Reference

- Original post: [https://x.com/feilsystem/status/2119841165491245507](https://x.com/feilsystem/status/2119841165491245507)
- Author: [@feilsystem](https://x.com/feilsystem) on X, posted 2026-11-09
