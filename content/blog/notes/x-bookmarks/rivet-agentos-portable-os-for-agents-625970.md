---
title: "Rivet agentOS Portable OS for Agents"
publishDate: "2026-04-01"
date: "2026-04-01"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-agents", "tools", "serving"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2039387320357625970-1.jpg"
excerpt: "Rivet launches agentOS, an open-source OS for agents using WASM and V8 isolates."
---

> Source: [@WesRoth on X](https://x.com/WesRoth/status/2039387320357625970) · posted 2026-04-01

![Rivet agentOS Portable OS for Agents (1)](/imgs/blogs/x-bookmarks/2039387320357625970-1.jpg)

![Rivet agentOS Portable OS for Agents (2)](/imgs/blogs/x-bookmarks/2039387320357625970-2.jpg)

## Summary

The post reports Rivet's beta of agentOS, an open-source OS for AI agents that embeds in a backend instead of using VMs or Docker. The quoted post claims about 6ms cold starts, 32x cheaper than sandboxes, and mounting S3 or SQLite as a file system.

## Background

Sandboxes isolate agent code. WASM and V8 isolates are lightweight alternatives to containers and start faster. The cost and speed figures are the vendor's claims.

## What I took away

- Lightweight isolates can cut cold-start cost.
- File-system mounts simplify agent data access.
- Verify the 32x claim for your workload.

## Reference

- Original post: [https://x.com/WesRoth/status/2039387320357625970](https://x.com/WesRoth/status/2039387320357625970)
- Author: [@WesRoth](https://x.com/WesRoth) on X, posted 2026-04-01
