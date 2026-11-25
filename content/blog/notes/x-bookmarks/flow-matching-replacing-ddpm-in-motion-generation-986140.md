---
title: "Flow Matching Replacing DDPM In Motion Generation"
publishDate: "2026-03-20"
date: "2026-03-20"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "learning-resources"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2035136752109986140-1.jpg"
excerpt: "The author argues flow matching is replacing DDPM in motion generation and points to MIT lecture notes that explain it."
---

> Source: [@CedricKuperman on X](https://x.com/CedricKuperman/status/2035136752109986140) · posted 2026-03-20

![Flow Matching Replacing DDPM In Motion Generation](/imgs/blogs/x-bookmarks/2035136752109986140-1.jpg)

## Summary

The author says flow matching is already replacing DDPM in motion generation, citing a cleaner formulation, fewer sampling steps and better scaling. They recommend learning it. The quoted post releases self-contained lecture notes on flow matching and diffusion from scratch, linked on arXiv.

## Background

DDPM is the classic diffusion model that generates data by removing noise over many steps. Flow matching trains a model to move noise toward data along a smooth path, which often needs fewer steps. The replacement claim is the author's view.

## What I took away

- Flow matching is worth learning if you work on generative models.
- The linked lecture notes are a free, from-scratch learning path.
- Fewer sampling steps means faster generation.

## Reference

- Original post: [https://x.com/CedricKuperman/status/2035136752109986140](https://x.com/CedricKuperman/status/2035136752109986140)
- Author: [@CedricKuperman](https://x.com/CedricKuperman) on X, posted 2026-03-20
