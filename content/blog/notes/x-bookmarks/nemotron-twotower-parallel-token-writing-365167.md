---
title: "Nemotron TwoTower Parallel Token Writing"
publishDate: "2026-07-01"
date: "2026-07-01"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "llm-architecture", "llm-inference"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2072402904867365167-1.jpg"
excerpt: "A post explains NVIDIA's TwoTower method, splitting a 30B model into a context reader and a parallel writer."
---

> Source: [@LiorOnAI on X](https://x.com/LiorOnAI/status/2072402904867365167) · posted 2026-07-01

![Nemotron TwoTower Parallel Token Writing](/imgs/blogs/x-bookmarks/2072402904867365167-1.jpg)

## Summary

@LiorOnAI says NVIDIA converted a 30B model into a faster one without retraining from scratch: duplicate the model, freeze one copy to read the prompt, and train the other to write. The quoted post introduces Nemotron-Labs-TwoTower, a diffusion language model adapted from Nemotron-3-Nano-30B-A3B.

## Background

Normal LLMs write one token at a time. A diffusion LM writes many in parallel. Splitting into a context tower and a writer tower lets the second model generate blocks.

## What I took away

- Existing models can be adapted to parallel decoding.
- Freezing one copy keeps context handling.
- Gains are the author's claim.

## Reference

- Original post: [https://x.com/LiorOnAI/status/2072402904867365167](https://x.com/LiorOnAI/status/2072402904867365167)
- Author: [@LiorOnAI](https://x.com/LiorOnAI) on X, posted 2026-07-01
