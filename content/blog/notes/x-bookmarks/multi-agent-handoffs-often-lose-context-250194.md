---
title: "Multi-Agent Handoffs Often Lose Context"
publishDate: "2026-08-03"
date: "2026-08-03"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-agents", "harness-engineering"]
author: "Hiep Tran"
featured: false
excerpt: "A post arguing two agents in a pipeline often do not collaborate: the second redoes work because handoffs pass too little context."
---

> Source: [@hanakoxbt on X](https://x.com/hanakoxbt/status/2084418314873250194) · posted 2026-08-03

## Summary

The author argues that in many two-agent setups the agents are not collaborating; the second agent redoes the first one's work. The trace looks like teamwork with clean handoffs, but agent A paid for six findings and agent B received only one sentence. The text is cut off as it starts to explain that two agents cannot share context.

## Background

In multi-agent systems, each agent has its own context window. A handoff is the message passed from one agent to the next. If the handoff is a short summary, the detail the first agent gathered is lost, so the second spends tokens rediscovering it.

## What I took away

- Check what actually passes between agents, not just that handoffs happen.
- Rich, structured handoffs can save repeated work.
- Sometimes one agent with full context beats two with thin handoffs.

## Reference

- Original post: [https://x.com/hanakoxbt/status/2084418314873250194](https://x.com/hanakoxbt/status/2084418314873250194)
- Author: [@hanakoxbt](https://x.com/hanakoxbt) on X, posted 2026-08-03
