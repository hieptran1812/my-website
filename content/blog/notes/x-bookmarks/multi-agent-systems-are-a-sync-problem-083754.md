---
title: "Multi-Agent Systems Are A Sync Problem"
publishDate: "2025-12-20"
date: "2025-12-20"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-agents", "harness-engineering"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2002187469786083754-1.jpg"
excerpt: "A short post argues multi-agent systems are a synchronization problem, quoting tldraw's lessons on coordinating agents on a canvas."
---

> Source: [@thruflo on X](https://x.com/thruflo/status/2002187469786083754) · posted 2025-12-20

![Multi-Agent Systems Are A Sync Problem](/imgs/blogs/x-bookmarks/2002187469786083754-1.jpg)

## Summary

The author states that multi-agent is a sync problem. The quoted post from tldraw says getting multiple AI agents to work together on an infinite canvas is a huge coordination problem and shares lessons from building their fairies feature.

## Background

When several agents change shared state at once, they need to see consistent data and avoid conflicting edits, much like collaborative editing. Sync tools and conflict handling matter as much as the agents themselves.

## What I took away

- Treat shared state between agents like a collaborative editing problem.
- Coordination design often matters more than model choice.

## Reference

- Original post: [https://x.com/thruflo/status/2002187469786083754](https://x.com/thruflo/status/2002187469786083754)
- Author: [@thruflo](https://x.com/thruflo) on X, posted 2025-12-20
