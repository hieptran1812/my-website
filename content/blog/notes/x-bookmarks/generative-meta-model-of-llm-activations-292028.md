---
title: "Generative Meta-Model of LLM Activations"
publishDate: "2026-02-15"
date: "2026-02-15"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "interpretability", "diffusion", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2023143347490292028-1.jpg"
excerpt: "Sarah Marks praises a paper that trains diffusion models on a billion LLM activations as an alternative to SAEs."
---

> Source: [@saprmarks on X](https://x.com/saprmarks/status/2023143347490292028) · posted 2026-02-15

![Generative Meta-Model of LLM Activations](/imgs/blogs/x-bookmarks/2023143347490292028-1.jpg)

## Summary

Marks says SAEs model LLM activations in an opinionated way, as sparse linear combinations. The quoted paper trains diffusion models on a billion LLM activations, in a less opinionated approach. She says it learns interpretable units with more natural causal effects on LLM computation.

## Background

Activations are the internal numbers a model computes. Sparse autoencoders (SAEs) break them into a few active features. A generative meta-model instead learns the distribution of activations directly, which may avoid assumptions built into SAEs.

## What I took away

- SAEs assume sparse linear structure; that is a choice, not a fact.
- Generative models of activations are a new tool for interpretability work.

## Reference

- Original post: [https://x.com/saprmarks/status/2023143347490292028](https://x.com/saprmarks/status/2023143347490292028)
- Author: [@saprmarks](https://x.com/saprmarks) on X, posted 2026-02-15
