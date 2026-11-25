---
title: "NVIDIA Nemotron 3 Release And Architecture Comparison"
publishDate: "2025-12-20"
date: "2025-12-20"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "open-models", "llm-architecture", "moe"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2002382751866188246-1.jpg"
excerpt: "Sebastian Raschka notes NVIDIA's Nemotron 3 series in three sizes and an updated LLM architecture comparison article."
---

> Source: [@rasbt on X](https://x.com/rasbt/status/2002382751866188246) · posted 2025-12-20

![NVIDIA Nemotron 3 Release And Architecture Comparison (1)](/imgs/blogs/x-bookmarks/2002382751866188246-1.jpg)

![NVIDIA Nemotron 3 Release And Architecture Comparison (2)](/imgs/blogs/x-bookmarks/2002382751866188246-2.jpg)

## Summary

Sebastian Raschka says he did not expect another major open-weight release in December, but NVIDIA released Nemotron 3 this week. It has Nano (30B-A3B), Super (100B) and Ultra (500B) sizes. He also quotes his updated Big LLM Architecture Comparison article, which has more than doubled since the July 2025 version.

## Background

Open-weight means the trained model files are public. The name 30B-A3B means about 30 billion total parameters with about 3 billion active per token, which is typical of mixture-of-experts models. Comparing architectures side by side helps engineers see which design choices recent models share.

## What I took away

- The A3B naming means active parameters are much fewer than total parameters.
- Use the architecture comparison article as a reference for recent open models.
- Open-weight releases keep arriving even late in the year.

## Reference

- Original post: [https://x.com/rasbt/status/2002382751866188246](https://x.com/rasbt/status/2002382751866188246)
- Author: [@rasbt](https://x.com/rasbt) on X, posted 2025-12-20
