---
title: "FNS Tokenizer With Character Resolution"
publishDate: "2026-06-09"
date: "2026-06-09"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "learning-resources", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2064290470600077701-1.jpg"
excerpt: "A post on FNS, a tokenizer that chunks efficiently but can decode single characters."
---

> Source: [@lvwerra on X](https://x.com/lvwerra/status/2064290470600077701) · posted 2026-06-09

![FNS Tokenizer With Character Resolution](/imgs/blogs/x-bookmarks/2064290470600077701-1.jpg)

## Summary

@lvwerra shares a deep dive into FNS. It adds a character-level signal to the loss during training so that at inference single characters can be decoded.

## Background

Tokenizers split text into chunks. Large chunks are efficient but hide characters, which hurts tasks like spelling.

## What I took away

- Tokenization limits character tasks.
- Read the deep dive.

## Reference

- Original post: [https://x.com/lvwerra/status/2064290470600077701](https://x.com/lvwerra/status/2064290470600077701)
- Author: [@lvwerra](https://x.com/lvwerra) on X, posted 2026-06-09
