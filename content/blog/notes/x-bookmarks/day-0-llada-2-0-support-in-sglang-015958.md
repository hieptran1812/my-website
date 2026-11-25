---
title: "Day-0 LLaDA 2.0 Support In SGLang"
publishDate: "2025-12-19"
date: "2025-12-19"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "serving", "post-training"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2002113332552015958-1.jpg"
excerpt: "LMSYS announces SGLang support for LLaDA 2.0 and a framework for diffusion LLM inference and RL post-training."
---

> Source: [@lmsysorg on X](https://x.com/lmsysorg/status/2002113332552015958) · posted 2025-12-19

![Day-0 LLaDA 2.0 Support In SGLang (1)](/imgs/blogs/x-bookmarks/2002113332552015958-1.jpg)

![Day-0 LLaDA 2.0 Support In SGLang (2)](/imgs/blogs/x-bookmarks/2002113332552015958-2.jpg)

## Summary

LMSYS says SGLang now supports LLaDA 2.0 from day zero in its dLLM framework. It notes that inference and RL post-training tools lag behind for diffusion LLMs. Compared with autoregressive models, dLLMs use more flexible and varied decoding strategies.

## Background

A diffusion LLM generates text by refining many token positions in parallel rather than one token at a time. That makes decoding more varied, so serving engines need new scheduling and sampling support. Good tooling is needed before these models can be used or trained at scale.

## What I took away

- Diffusion LLMs need different serving logic than autoregressive models.
- Day-0 framework support lowers the cost of trying a new model.
- RL post-training for dLLMs is still an immature area.

## Reference

- Original post: [https://x.com/lmsysorg/status/2002113332552015958](https://x.com/lmsysorg/status/2002113332552015958)
- Author: [@lmsysorg](https://x.com/lmsysorg) on X, posted 2025-12-19
