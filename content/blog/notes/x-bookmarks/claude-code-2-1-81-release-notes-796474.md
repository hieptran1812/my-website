---
title: "Claude Code 2.1.81 Release Notes"
publishDate: "2026-03-20"
date: "2026-03-20"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "tools", "ai-agents"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2035133196115796474-1.jpg"
excerpt: "A changelog bot summarizes Claude Code 2.1.81, including a new --bare flag and a no-memory behavior."
---

> Source: [@ClaudeCodeLog on X](https://x.com/ClaudeCodeLog/status/2035133196115796474) · posted 2026-03-20

![Claude Code 2.1.81 Release Notes](/imgs/blogs/x-bookmarks/2035133196115796474-1.jpg)

## Summary

The post says Claude Code 2.1.81 has 27 CLI changes and 2 system prompt changes. Highlights include a --bare flag that skips hooks, LSP, plugin sync and skill scans for faster, deterministic automation, and the assistant now honors 'no memory' requests by ignoring stored memory. The text is cut off.

## Background

Hooks, LSP (language server) and plugins add startup work. A bare mode helps scripts and CI run faster and more predictably. Honoring no-memory requests helps with clean, reproducible runs.

## What I took away

- Use --bare for scripted or CI runs that need speed and determinism.
- Read changelogs to catch flags that affect automation.

## Reference

- Original post: [https://x.com/ClaudeCodeLog/status/2035133196115796474](https://x.com/ClaudeCodeLog/status/2035133196115796474)
- Author: [@ClaudeCodeLog](https://x.com/ClaudeCodeLog) on X, posted 2026-03-20
