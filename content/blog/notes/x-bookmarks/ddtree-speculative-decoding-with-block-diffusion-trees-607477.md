---
title: "DDTree Speculative Decoding With Block Diffusion Trees"
publishDate: "2026-04-13"
date: "2026-04-13"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "speculative-decoding", "diffusion", "llm-inference"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2043813397972607477-1.jpg"
excerpt: "DDTree drafts a tree of continuations in one block diffusion pass and verifies several at once."
---

> Source: [@liranringel on X](https://x.com/liranringel/status/2043813397972607477) · posted 2026-04-13

![DDTree Speculative Decoding With Block Diffusion Trees](/imgs/blogs/x-bookmarks/2043813397972607477-1.jpg)

## Summary

The author introduces **DDTree**. It speeds up speculative decoding by drafting a tree with one block diffusion pass and then verifying multiple likely continuations together. Paper, project page and code are linked.

## Background

Normal speculative decoding checks one drafted line. A tree holds several options, so more tokens can be accepted per check. Block diffusion produces a block of tokens at once.

## What I took away

- Trees increase accepted tokens.
- Drafting in one pass is cheap.
- Read the paper for numbers.

## Reference

- Original post: [https://x.com/liranringel/status/2043813397972607477](https://x.com/liranringel/status/2043813397972607477)
- Author: [@liranringel](https://x.com/liranringel) on X, posted 2026-04-13
