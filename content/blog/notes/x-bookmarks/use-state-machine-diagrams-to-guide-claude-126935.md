---
title: "Use State Machine Diagrams to Guide Claude"
publishDate: "2026-01-10"
date: "2026-01-10"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-agents", "harness-engineering", "tools"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2009986072953126935-1.png"
excerpt: "A vibecoding tip: ask Claude for state machine diagrams of existing components to force full path coverage."
---

> Source: [@mert on X](https://x.com/mert/status/2009986072953126935) · posted 2026-01-10

![Use State Machine Diagrams to Guide Claude](/imgs/blogs/x-bookmarks/2009986072953126935-1.png)

## Summary

The author says that asking Opus 4.5 to make state machine diagrams of existing components makes it map out all paths. Otherwise, they say, it defaults to being lazy. The diagrams also help you verify that it is doing things correctly at a systems level.

## Background

A state machine diagram shows the states a component can be in and the transitions between them. Forcing the model to draw it makes it enumerate cases it might skip. It also gives you something quick to check against your own understanding.

## What I took away

- Ask the model for a diagram before asking it to change code.
- Diagrams make hidden branches and missing cases visible.
- Use them as a review aid for your own checks.

## Reference

- Original post: [https://x.com/mert/status/2009986072953126935](https://x.com/mert/status/2009986072953126935)
- Author: [@mert](https://x.com/mert) on X, posted 2026-01-10
