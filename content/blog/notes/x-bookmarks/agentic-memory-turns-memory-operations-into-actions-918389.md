---
title: "Agentic Memory Turns Memory Operations Into Actions"
publishDate: "2026-03-07"
date: "2026-03-07"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-agents", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2030338412230918389-1.jpg"
excerpt: "A summary of Alibaba and WHU work that lets the model manage its own memory through add and delete actions."
---

> Source: [@louszbd on X](https://x.com/louszbd/status/2030338412230918389) · posted 2026-03-07

![Agentic Memory Turns Memory Operations Into Actions](/imgs/blogs/x-bookmarks/2030338412230918389-1.jpg)

## Summary

The post describes new work from Alibaba and WHU called Agentic Memory. Most agent memory today is hardcoded infrastructure: a vector database plus hand-written rules for when to store, delete or summarize. The new approach makes memory operations into actions, such as add and delete, that the model itself can choose.

## Background

A vector database stores text as numeric embeddings so it can be searched by meaning. Agents usually rely on fixed rules around it. Letting the model call memory operations like tools means it can decide what to keep, which may adapt better but is harder to control.

## What I took away

- Treat memory operations as tools the agent can call.
- Hand-written storage rules can become a bottleneck.
- Model-managed memory raises questions about reliability and evaluation.

## Reference

- Original post: [https://x.com/louszbd/status/2030338412230918389](https://x.com/louszbd/status/2030338412230918389)
- Author: [@louszbd](https://x.com/louszbd) on X, posted 2026-03-07
