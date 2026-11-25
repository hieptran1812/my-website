---
title: "LoopRes Loops Pretrained LLMs for Gains"
publishDate: "2026-11-23"
date: "2026-11-23"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "post-training", "open-models"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2124890218373534069-1.jpg"
excerpt: "The authors introduce LoopRes, a method to loop any pretrained LLM and report up to 37.7% higher scores than SFT."
---

> Source: [@oahzxl on X](https://x.com/oahzxl/status/2124890218373534069) · posted 2026-11-23

![LoopRes Loops Pretrained LLMs for Gains](/imgs/blogs/x-bookmarks/2124890218373534069-1.jpg)

## Summary

The author presents LoopRes, which loops any pretrained LLM for what they describe as significant and consistent gains. They claim up to 37.7% higher scores than SFT, validated on 6 LLMs from 3 families up to 30B parameters, with code, data and models open-sourced.

## Background

A looped model reuses the same layers several times, giving it more computation per token without adding parameters. Applying this to an already trained model would be a cheap way to boost reasoning. The numbers are the authors' own claims.

## What I took away

- Looping reuses layers to add compute without more weights.
- The 37.7% figure is the authors' claim against SFT.
- Code and models are open, so the result can be checked.

## Reference

- Original post: [https://x.com/oahzxl/status/2124890218373534069](https://x.com/oahzxl/status/2124890218373534069)
- Author: [@oahzxl](https://x.com/oahzxl) on X, posted 2026-11-23
