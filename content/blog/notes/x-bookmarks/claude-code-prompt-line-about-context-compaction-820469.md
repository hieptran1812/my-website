---
title: "Claude Code Prompt Line About Context Compaction"
publishDate: "2025-11-23"
date: "2025-11-23"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-agents", "harness-engineering", "tools"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1992478858025820469-1.png"
excerpt: "A tip to add a line to the system prompt telling Claude not to stop early because of token budget concerns."
---

> Source: [@kieranklaassen on X](https://x.com/kieranklaassen/status/1992478858025820469) · posted 2025-11-23

![Claude Code Prompt Line About Context Compaction](/imgs/blogs/x-bookmarks/1992478858025820469-1.png)

## Summary

The author suggests adding a line to the Claude Code system prompt. It says the context window will be automatically compacted as it nears the limit, and that the model should never stop tasks early due to token budget concerns and should always complete them fully.

## Background

Context compaction means older conversation content is summarized so work can continue past the context limit. Models sometimes wrap up early if they think they are running out of room. Telling them compaction exists can reduce that behavior. The linked card points to prompting best practices.

## What I took away

- Tell the model about compaction so it does not rush to finish.
- Small system prompt lines can change agent behavior on long tasks.
- Test the effect on your own tasks before relying on it.

## Reference

- Original post: [https://x.com/kieranklaassen/status/1992478858025820469](https://x.com/kieranklaassen/status/1992478858025820469)
- Author: [@kieranklaassen](https://x.com/kieranklaassen) on X, posted 2025-11-23
