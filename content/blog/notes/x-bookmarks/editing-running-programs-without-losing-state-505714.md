---
title: "Editing Running Programs Without Losing State"
publishDate: "2025-12-04"
date: "2025-12-04"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "tools", "harness-engineering"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1996646914721505714-1.jpg"
excerpt: "The author argues that live editing of running programs without losing state could save huge amounts of developer time."
---

> Source: [@io_sammt on X](https://x.com/io_sammt/status/1996646914721505714) · posted 2025-12-04

![Editing Running Programs Without Losing State (1)](/imgs/blogs/x-bookmarks/1996646914721505714-1.jpg)

![Editing Running Programs Without Losing State (2)](/imgs/blogs/x-bookmarks/1996646914721505714-2.jpg)

## Summary

The author claims that editing a program while it runs, without losing any state, could save humanity at least 250 billion hours per year. Their motto is never recompile, reload or redeploy. The quoted post describes the **Unit Environment**, where you interact with an app and debug its internals in the same place, so each project can have its own developer experience.

## Background

Normally a code change means rebuilding and restarting, which throws away in-memory state. Live programming systems, like Lisp REPLs, let you change code in place. The 250 billion hours figure is the author's claim.

## What I took away

- Restart cycles are a hidden cost in development.
- Live-editing environments keep state while you change code.
- Treat the savings figure as the author's rough estimate.

## Reference

- Original post: [https://x.com/io_sammt/status/1996646914721505714](https://x.com/io_sammt/status/1996646914721505714)
- Author: [@io_sammt](https://x.com/io_sammt) on X, posted 2025-12-04
