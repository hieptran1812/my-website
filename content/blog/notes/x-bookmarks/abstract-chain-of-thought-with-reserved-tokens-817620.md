---
title: "Abstract Chain-of-Thought With Reserved Tokens"
publishDate: "2026-04-27"
date: "2026-04-27"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "ai-research", "llm-inference"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2048743883580817620-1.jpg"
excerpt: "A new method lets language models reason through a short sequence of reserved abstract tokens, trained with reinforcement learning."
---

> Source: [@KeshavRamji on X](https://x.com/KeshavRamji/status/2048743883580817620) · posted 2026-04-27

![Abstract Chain-of-Thought With Reserved Tokens](/imgs/blogs/x-bookmarks/2048743883580817620-1.jpg)

## Summary

The authors introduce **Abstract Chain-of-Thought**. It lets a language model reason through a short sequence of reserved "abstract" tokens, learned with reinforcement learning. The post asks whether a model could reason efficiently in an entirely new language.

## Background

Chain-of-thought means the model writes out reasoning steps before the answer, which costs many tokens. Here the model instead uses a small set of special tokens as a compact reasoning code. Shorter reasoning could make answers cheaper and faster.

## What I took away

- Reasoning does not have to be in readable English.
- Reinforcement learning can shape a compact reasoning format.
- Check the paper for how accuracy and token cost compare; the post text is cut off.

## Reference

- Original post: [https://x.com/KeshavRamji/status/2048743883580817620](https://x.com/KeshavRamji/status/2048743883580817620)
- Author: [@KeshavRamji](https://x.com/KeshavRamji) on X, posted 2026-04-27
