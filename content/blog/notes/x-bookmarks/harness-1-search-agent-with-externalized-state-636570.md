---
title: "Harness-1 Search Agent With Externalized State"
publishDate: "2026-06-06"
date: "2026-06-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-agents", "harness-engineering", "post-training"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2063298457398636570-1.jpg"
excerpt: "Introduces Harness-1, a 20B search agent trained with a harness that stores search state outside the model."
---

> Source: [@patpcj on X](https://x.com/patpcj/status/2063298457398636570) · posted 2026-06-06

![Harness-1 Search Agent With Externalized State](/imgs/blogs/x-bookmarks/2063298457398636570-1.jpg)

## Summary

The author introduces Harness-1, a 20B search agent trained with a state-externalizing harness. They claim frontier-level long-horizon search, rivaling Opus-4.6 and beating GPT-5.4, at Context-1-level cost and latency. The harness externalizes candidates, evidence, verification and search history.

## Background

A harness is the code around a model that manages tools, memory and loops. Externalizing state means keeping things like evidence and history in the harness instead of in the model's context window. This can keep long searches manageable for a small model.

## What I took away

- Moving state out of the context can help small models do long tasks.
- These performance numbers are the author's claims; check the details.
- Harness design matters as much as model size for agents.

## Reference

- Original post: [https://x.com/patpcj/status/2063298457398636570](https://x.com/patpcj/status/2063298457398636570)
- Author: [@patpcj](https://x.com/patpcj) on X, posted 2026-06-06
