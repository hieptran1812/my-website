---
title: "Two Kimi Papers Every LLM Engineer Should Read"
publishDate: "2026-07-24"
date: "2026-07-24"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "The author recommends Kimi's papers on Kimi Delta Attention and Attention Residuals for LLM engineers."
---

> Source: [@Alacritic_Super on X](https://x.com/Alacritic_Super/status/2080561371662426569) · posted 2026-07-24

## Summary

The author recommends two papers from the Kimi team. The first, **Kimi Delta Attention**, uses a linear attention mechanism to improve long-context efficiency and reduce KV cache usage, with an arXiv link. The second is about Attention Residuals (text cut off).

## Background

Linear attention replaces the standard attention that grows with sequence length with a fixed-size running state, saving memory on long inputs. The KV cache is the stored attention data that grows with context. These ideas underlie Kimi K3's long-context speed claims.

## What I took away

- Read the KDA paper to understand linear attention in a frontier model.
- Reducing KV cache size is a main path to cheaper long-context inference.
- Look up the Attention Residuals paper for the second idea.

## Reference

- Original post: [https://x.com/Alacritic_Super/status/2080561371662426569](https://x.com/Alacritic_Super/status/2080561371662426569)
- Author: [@Alacritic_Super](https://x.com/Alacritic_Super) on X, posted 2026-07-24
