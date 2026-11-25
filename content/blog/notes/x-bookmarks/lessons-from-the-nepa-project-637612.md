---
title: "Lessons From The NEPA Project"
publishDate: "2025-12-19"
date: "2025-12-19"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2002101201404637612-1.jpg"
excerpt: "An author shares what he learned from NEPA, starting with representation learning emerging from generative objectives."
---

> Source: [@wenhaocha1 on X](https://x.com/wenhaocha1/status/2002101201404637612) · posted 2025-12-19

![Lessons From The NEPA Project](/imgs/blogs/x-bookmarks/2002101201404637612-1.jpg)

## Summary

The author lists three lessons from the project, first that representation learning can fully emerge from generative objectives. In language this is nearly a consensus, but in vision discriminative methods such as CLIP and DINO still dominate. The quoted post introduces NEPA, which predicts the next embedding from all previous embeddings.

## Background

A generative objective trains a model to produce data, while a discriminative one trains it to tell things apart. Language models learn good representations from next-token prediction. NEPA tests whether the same holds for vision. Only the first lesson is shown.

## What I took away

- Generative training can yield good representations in vision too.
- CLIP and DINO are the discriminative baselines to compare against.

## Reference

- Original post: [https://x.com/wenhaocha1/status/2002101201404637612](https://x.com/wenhaocha1/status/2002101201404637612)
- Author: [@wenhaocha1](https://x.com/wenhaocha1) on X, posted 2025-12-19
