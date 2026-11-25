---
title: "Google Research Neural Memory And Test-Time Training"
publishDate: "2025-12-06"
date: "2025-12-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1997227157748060277-1.jpg"
excerpt: "A summary of two Google Research papers on architectures that update their own parameters during inference."
---

> Source: [@_philschmid on X](https://x.com/_philschmid/status/1997227157748060277) · posted 2025-12-06

![Google Research Neural Memory And Test-Time Training](/imgs/blogs/x-bookmarks/1997227157748060277-1.jpg)

## Summary

@_philschmid asks what comes after Transformers and answers with Neural Memory and Test-Time Training. @GoogleResearch presented two new papers at NeurIPS with an architecture that actively learns and updates its own parameters during inference, acting as a long-term neural memory rather than a static model.

## Background

A standard transformer's weights are fixed once deployed, and it only reads its context window. Test-time training lets part of the model learn from the input as it is processed, which works like memory. The aim is to handle very long context efficiently.

## What I took away

- Models that learn at inference time are an active research direction.
- Long-term memory is a key limit of current transformers.
- This is a research direction, not a replacement that is proven in production.

## Reference

- Original post: [https://x.com/_philschmid/status/1997227157748060277](https://x.com/_philschmid/status/1997227157748060277)
- Author: [@_philschmid](https://x.com/_philschmid) on X, posted 2025-12-06
