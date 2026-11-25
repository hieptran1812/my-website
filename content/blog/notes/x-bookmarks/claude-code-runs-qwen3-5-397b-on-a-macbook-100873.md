---
title: "Claude Code Runs Qwen3.5-397B On A MacBook"
publishDate: "2026-03-17"
date: "2026-03-17"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "ai-agents", "hardware"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2033940227736100873-1.png"
excerpt: "The author says Claude Code, given autoresearch and Apple's LLM in a Flash paper, got Qwen3.5-397B running on an M3 Max 48GB."
---

> Source: [@danveloper on X](https://x.com/danveloper/status/2033940227736100873) · posted 2026-03-17

![Claude Code Runs Qwen3.5-397B On A MacBook](/imgs/blogs/x-bookmarks/2033940227736100873-1.png)

## Summary

The author says they handed Claude Code Karpathy's autoresearch repo and Apple's LLM in a Flash paper, asked it to run Qwen3.5-397B on an M3 Max with 48GB, and it did.

## Background

The LLM in a Flash approach keeps weights in flash storage and loads only what is needed. Combined with an automated experiment loop, an agent can tune such a setup itself. The post gives no speed numbers.

## What I took away

- Agents can implement research papers when given a clear goal and a metric.
- Ask for tokens per second to judge practicality.

## Reference

- Original post: [https://x.com/danveloper/status/2033940227736100873](https://x.com/danveloper/status/2033940227736100873)
- Author: [@danveloper](https://x.com/danveloper) on X, posted 2026-03-17
