---
title: "Ten Ways To Stop Wasting Coding Tokens"
publishDate: "2026-05-12"
date: "2026-05-12"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-agents", "harness-engineering", "tools"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2054255152555545079-1.jpg"
excerpt: "A post attributes to Karpathy the claim that most AI coding cost is unneeded context and lists ten token-saving habits."
---

> Source: [@DeRonin_ on X](https://x.com/DeRonin_/status/2054255152555545079) · posted 2026-05-12

![Ten Ways To Stop Wasting Coding Tokens](/imgs/blogs/x-bookmarks/2054255152555545079-1.jpg)

## Summary

The post attributes to Andrej Karpathy the quote that 90% of an AI coding bill pays for context you did not need to send. It lists ten things senior engineers stopped wasting tokens on. The first example is auto-loading 50 files for a 30-line fix, which the author says costs $1.20 per turn with 80% input waste.

## Background

Tokens are the units LLMs read and write, and API cost scales with them. Coding agents often send large amounts of files as context each turn. Sending only relevant files cuts cost and often improves focus. The quote and numbers are the post author's claims.

## What I took away

- Send only the files the task needs, not the whole repo.
- Context cost repeats every turn, so waste adds up.
- Verify attributed quotes before relying on them.

## Reference

- Original post: [https://x.com/DeRonin_/status/2054255152555545079](https://x.com/DeRonin_/status/2054255152555545079)
- Author: [@DeRonin_](https://x.com/DeRonin_) on X, posted 2026-05-12
