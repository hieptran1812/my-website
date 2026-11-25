---
title: "Tiny Diffusion Shrunk To 364 Lines"
publishDate: "2025-12-30"
date: "2025-12-30"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "llm-architecture", "learning-resources"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2006075828602614115-1.jpg"
excerpt: "A rewrite cuts a tiny text diffusion model to 364 lines in one file and adds a tiny GPT for comparison."
---

> Source: [@nathanrs on X](https://x.com/nathanrs/status/2006075828602614115) · posted 2025-12-30

![Tiny Diffusion Shrunk To 364 Lines (1)](/imgs/blogs/x-bookmarks/2006075828602614115-1.jpg)

![Tiny Diffusion Shrunk To 364 Lines (2)](/imgs/blogs/x-bookmarks/2006075828602614115-2.jpg)

## Summary

Nathan rewrote tiny-diffusion to be 3x smaller, from 951 lines to 364 in a single file. He also added a tiny GPT implementation of 312 lines for comparison, and the two are about 80% identical. The quoted post describes the original, an 11M parameter character-level text diffusion model based on modified nanochat and trained on Tiny Shakespeare.

## Background

A text diffusion model generates text by gradually refining noisy or masked tokens, instead of predicting one token after another as GPT does. Having both models in near-identical code makes the real differences easy to see.

## What I took away

- Small code makes the difference between diffusion and autoregressive models easy to study.
- Side-by-side implementations are a good learning tool.

## Reference

- Original post: [https://x.com/nathanrs/status/2006075828602614115](https://x.com/nathanrs/status/2006075828602614115)
- Author: [@nathanrs](https://x.com/nathanrs) on X, posted 2025-12-30
