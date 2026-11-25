---
title: "Interleaved Thinking Explains New Agent APIs"
publishDate: "2025-12-16"
date: "2025-12-16"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-agents", "harness-engineering", "open-models"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2000991077549961342-1.jpg"
excerpt: "Niels Rogge says OpenAI's Responses API and Google's Interactions API exist because of interleaved thinking, pioneered by MiniMax."
---

> Source: [@NielsRogge on X](https://x.com/NielsRogge/status/2000991077549961342) · posted 2025-12-16

![Interleaved Thinking Explains New Agent APIs](/imgs/blogs/x-bookmarks/2000991077549961342-1.jpg)

## Summary

The author asks why OpenAI recommends the Responses API and Google DeepMind built the Interactions API. His answer is interleaved thinking, which he says was pioneered by MiniMax for open source and leads to more reliable agents. He starts a thread to explain.

## Background

Interleaved thinking means the model's reasoning is kept and reused between tool calls rather than dropped after each step. APIs that preserve this reasoning state let agents stay consistent across many steps. Older chat APIs that only pass messages can lose it.

## What I took away

- Keep reasoning state between tool calls to make agents more reliable.
- API design affects whether reasoning is preserved.
- Check whether your agent framework passes reasoning back to the model.

## Reference

- Original post: [https://x.com/NielsRogge/status/2000991077549961342](https://x.com/NielsRogge/status/2000991077549961342)
- Author: [@NielsRogge](https://x.com/NielsRogge) on X, posted 2025-12-16
